import torch
from collections.abc import Mapping
from torch import nn
from tqdm import tqdm
import copy
import json

from llm_nav.dataset import preprocess
from llm_nav.utils import make_evaluation_data_module

from llm_nav.sim.planner import RelativeGreedyAgent
from llm_nav.data import FeatureDB


class FLAME:
    def __init__(self, opts, sna_module=None):
        self.max_route_len = opts.max_route_len
        self.use_feature = opts.store_feature
        self.legacy_navigation_mode = opts.legacy_navigation_mode
        self.temperature = opts.temperature
        self.decoding_paths = opts.decoding_paths
        self.autocast = torch.cuda.amp.autocast
        self.sna = sna_module
        self.sna_memory = {}
        self._log("[Init] Global SNA Memory initialized.")
        db_raw_path = getattr(opts, 'img_raw_db', None)
        db_insert_path = getattr(opts, 'img_raw_insert_db', None)
        insert_task_dataset = getattr(opts, 'insert_task_dataset', None)
        if db_raw_path and FeatureDB is not None:
            self._log(f"[Init] Loading FeatureDB from: {db_raw_path} ...")
            self.feature_db = FeatureDB(insert_dir=db_insert_path, default_dir=db_raw_path,
                                        insert_task_dataset=insert_task_dataset, decode_type='pil')
            self._log(f"[Init] FeatureDB loaded. Keys: {len(self.feature_db.map)}")
        else:
            self._log(f"[Init] Warning: 'img_raw_db' not set in opts or FeatureDB class missing.")
            self._log(f"       Smart Detour (Planner) will be DISABLED.")
            self.feature_db = None

        if opts.insert_task_dataset is not None:
            with open(opts.insert_task_dataset, 'r', encoding='utf-8') as f:
                self.insert_data = json.load(f)
        else:
            self.insert_data = None

    def _log(self, message):
        print(message)

    def _prepare_input(self, data, model):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v, model) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v, model) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": model.device}
            return data.to(**kwargs, non_blocking=True)
        return data

    def _make_batch(self, model, conv_batch, tokenizer, input_ids=None):
        sources = copy.deepcopy([e["conversations"] for e in conv_batch])
        data_dict = preprocess(sources, tokenizer, inference_mode=True, concat_mode=input_ids is not None,
                               task="instruction_following")
        images_per_example = max(len(e['images']) for e in conv_batch)
        if self.use_feature:
            batch_images = [torch.stack(item['images']) for item in conv_batch]
            batch_images = nn.utils.rnn.pad_sequence(batch_images, batch_first=True)
            batch_images = batch_images.unsqueeze(2)
        else:
            batch_images = None
            for iexample, example in enumerate(conv_batch):
                for iimage, image in enumerate(example['images']):
                    preprocessed = torch.from_numpy(image)
                    if batch_images is None:
                        batch_images = torch.zeros(
                            (len(conv_batch), images_per_example, 1) + preprocessed.shape,
                            dtype=preprocessed.dtype,
                        )
                    batch_images[iexample, iimage, 0] = preprocessed

        reversed_input_ids = [torch.flip(ids, [0]) for ids in data_dict['input_ids']]
        padded_reversed_input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        new_input_ids = torch.flip(padded_reversed_input_ids, [1])

        input_ids = torch.cat([input_ids, new_input_ids.to(input_ids.device)],
                              dim=1) if input_ids is not None else new_input_ids
        batch = dict(
            lang_x=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            vision_x=batch_images,
        )

        return batch

    def _get_action(self, response):
        if not response.endswith('.'):
            true_response = response.split('.')[-1]
        else:
            true_response = response.split('.')[-2]

        lower_resp = true_response.lower()
        if 'stop' in lower_resp:
            action = 'stop'
        elif 'turn around' in lower_resp:
            action = 'turn_around'
        elif 'forward' in lower_resp or 'straight' in lower_resp:
            action = 'forward'
        elif 'left' in lower_resp:
            action = 'left'
        elif 'right' in lower_resp:
            action = 'right'
        else:
            action = 'unk'
        return action

    def self_consistency(self, response_batch, outputs):
        actions = []
        for response in response_batch:
            action = self._get_action(response)
            actions.append(action)
        action_count = {action: 0 for action in actions}
        for i, action in enumerate(actions):
            action_count[action] += 1
        max_action = max(action_count, key=action_count.get)
        for i, action in enumerate(actions):
            if action == max_action:
                max_action_index = i
                break
        outputs['past_key_values'] = [(past_key_value[0][max_action_index].unsqueeze(0),
                                       past_key_value[1][max_action_index].unsqueeze(0)) for past_key_value in
                                      outputs['past_key_values']]
        outputs['sequences'] = outputs['sequences'][max_action_index].unsqueeze(0)
        return [response_batch[max_action_index]], outputs['sequences'], outputs['past_key_values']

    def _sync_eval_traj_with_navigator(self, trajs, agent_idx, navigator):
        pano_path = getattr(navigator, "pano_path", None)
        if pano_path and 0 <= agent_idx < len(trajs):
            trajs[agent_idx] = list(pano_path)

    def _run_detour_task(self, env, model, tokenizer, agent_idx, sna_instruction, proposed_action,
                         ended_global, num_act_nav, trajs, total_steps):
        """
        """
        if self.feature_db is None or RelativeGreedyAgent is None:
            return

        actual_action = sna_instruction
        if actual_action == 'unk':
            return

        sim_agent = env.env.navs[agent_idx]
        curr_viewpoint, curr_heading_deg = sim_agent.get_state()
        if curr_viewpoint is None:
            return

        states_backup = list(sim_agent.states)


        planner = RelativeGreedyAgent(sim_agent, self.feature_db)

        planner.set_goal_by_expectation(curr_viewpoint, curr_heading_deg, proposed_action)

        complete_actions = planner.generate_detour_sequence(curr_viewpoint, curr_heading_deg, actual_action,proposed_action)

        sim_agent.states = states_backup

        if complete_actions == "fail_detour":
            return "fail_detour"
        elif complete_actions == []:
            return False
        else:
            for step_idx, action in enumerate(complete_actions):
                current_vp, current_heading = sim_agent.get_state()

                if ended_global[agent_idx]:
                    break


                full_actions = ['<detour_skip>'] * len(ended_global)
                full_actions[agent_idx] = action

                env.env.action_step(full_actions, ended_global, num_act_nav, trajs, total_steps, step_model=True)

            self._sync_eval_traj_with_navigator(trajs, agent_idx, sim_agent)
            return True

    @torch.inference_mode()
    def rollout(self, env, model, tokenizer, route_ids=None):
        trajs = env.reset(route_ids=route_ids)
        batch_size = len(route_ids)
        ended = [0] * batch_size
        num_act_nav = [batch_size]
        total_steps = [0]
        conv_batch = [{"route_id": item['route_id'] if "route_id" in item else item['id']} for item in env.batch]

        if env.env.name.startswith("touchdown"):
            init_msg = "You are playing the role of an agent. based on the navigation_text, view the image and decide the next action from ['Forward', 'Left', 'Right', 'Turn Around', 'Stop']."
            init_msg_legacy = init_msg
            begin_msg = "Turn around or head forward?"
            begin_msg_legacy = begin_msg
            user_msg = "Forward or stop?"
            user_msg_legacy = "Head forward or stop?"
        elif env.env.name.startswith("map2seq"):
            init_msg = "You are playing the role of an agent. based on the navigation_text, view the image and decide the next action from ['Forward', 'Left', 'Right', 'Turn Around', 'Stop']."
            init_msg_legacy = init_msg
            begin_msg = "Forward or stop?"
            begin_msg_legacy = "Head forward or stop?"
            user_msg = "Forward or stop?"
            user_msg_legacy = "Head forward or stop?"
        else:
            raise ValueError(f"Unknown env name: {env.env.name}")

        input_ids = None
        past_key_values = None
        responses = [[] for _ in range(batch_size)]

        for step in range(self.max_route_len):
            observations = env.get_observations()

            for i in range(batch_size):
                obs = observations[i]
                viewpoint = obs['viewpoint']
                image = obs['image']
                if ended[i]:
                    conv_batch[i]['conversations'] = [{"from": "User", "value": "", "num_images": 0}]
                    continue
                if step == 0:
                    conv_batch[i]['obs'] = [viewpoint]
                else:
                    conv_batch[i]['obs'].append(viewpoint)

                vp = viewpoint[:viewpoint.rfind('_')]
                neighbours_num = env.env.navs[i].graph.get_num_neighbors(vp)

                if step == 0:
                    conv_batch[i]['images'] = [image]
                    if self.legacy_navigation_mode:
                        init = f"{init_msg_legacy} navigation_text: {env.batch[i]['navigation_text']}"
                        begin = begin_msg_legacy
                    else:
                        init = f"{init_msg} navigation_text: {env.batch[i]['navigation_text']}"
                        begin = begin_msg
                    conv_batch[i]['conversations'] = [{"from": "User", "value": init, "num_images": 0},
                                                      {"from": "User", "value": begin, "num_images": 1}]
                elif 'intersection' in obs:
                    conv_batch[i]['images'].append(image)
                    if self.legacy_navigation_mode:
                        user = f"You've arrived at a {neighbours_num}-way intersection. Head forward, turn left, turn right or stop?"
                    else:
                        user = f"You've arrived at a {neighbours_num}-way intersection. Consider your next move."
                    conv_batch[i]['conversations'] = [{"from": "User", "value": user, "num_images": 1}]
                else:
                    user = user_msg_legacy if self.legacy_navigation_mode else user_msg
                    conv_batch[i]['images'].append(image)
                    conv_batch[i]['conversations'] = [{"from": "User", "value": user, "num_images": 1}]

            batch = self._make_batch(model, conv_batch, tokenizer, input_ids=input_ids)
            batch = self._prepare_input(batch, model)
            batch["past_key_values"] = past_key_values

            with self.autocast():
                if len(conv_batch) == 1 and 'intersection' in observations[0]:
                    outputs = model.generate_lightning(**batch, ended=ended, max_new_tokens=500,
                                                       num_return_sequences=self.decoding_paths,
                                                       temperature=self.temperature)
                else:
                    outputs = model.generate_lightning(**batch, ended=ended, max_new_tokens=500, num_return_sequences=1,
                                                       temperature=0.0)
            input_ids = outputs['sequences']
            past_key_values = outputs['past_key_values']

            response_batch_ids = outputs['sequences'][:, len(batch["lang_x"][0]):]
            response_batch = tokenizer.batch_decode(response_batch_ids, skip_special_tokens=True)

            if len(conv_batch) == 1 and (len(conv_batch) != len(response_batch)):
                response_batch, input_ids, past_key_values = self.self_consistency(response_batch, outputs)

            sna_batch_inputs = []
            sna_batch_indices = []  # Keep track of which agent corresponds to which SNA input
            proposed_actions_list = []  # Store proposed actions for later


            sna_results_map = {}

            for i, response in enumerate(response_batch):
                responses[i].append(response)
                if ended[i]:
                    proposed_actions_list.append("stop")
                    continue

                proposed_action = self._get_action(response)
                proposed_actions_list.append(proposed_action)

                if self.sna:
                    current_vp_id = observations[i]['viewpoint']
                    memory_key = (current_vp_id, proposed_action)

                    if memory_key in self.sna_memory:
                        sna_results_map[i] = self.sna_memory[memory_key]
                        continue

                    raw_img_obj = observations[i].get('raw_image')
                    sna_viewpoint = observations[i]['viewpoint']
                    if raw_img_obj is None:
                        raise RuntimeError(
                            "SNA requires raw 120-degree images. Set img_raw_db "
                            "and img_raw_insert_db to valid image stores."
                        )

                    vp = observations[i]['viewpoint'][:observations[i]['viewpoint'].rfind('_')]
                    neighbours_num = env.env.navs[i].graph.get_num_neighbors(vp)
                    if neighbours_num <= 2:
                        road_type = "2-way street"
                    else:
                        road_type = f"{neighbours_num}-way intersection"

                    sna_batch_inputs.append({
                        'image': raw_img_obj,
                        'proposed_action': proposed_action,
                        'road_type': road_type,
                        'viewpoint': sna_viewpoint
                    })
                    sna_batch_indices.append(i)


            if self.sna and len(sna_batch_inputs) > 0:
                if hasattr(self.sna, 'run_batch'):
                    batch_results = self.sna.run_batch(sna_batch_inputs)
                else:
                    batch_results = []
                    for inp in sna_batch_inputs:
                        res = self.sna.run(inp['image'], inp['proposed_action'], inp['road_type'], inp['viewpoint'])
                        batch_results.append(res)

                for idx, result in zip(sna_batch_indices, batch_results):
                    sna_results_map[idx] = result

                    cur_vp = observations[idx]['viewpoint']
                    cur_act = proposed_actions_list[idx]
                    mem_key = (cur_vp, cur_act)

                    self.sna_memory[mem_key] = result

            final_actions = []

            for i in range(batch_size):
                if ended[i]:
                    final_actions.append("stop")
                    continue

                proposed_action = proposed_actions_list[i]
                action_to_execute = proposed_action

                if i in sna_results_map:
                    raw_response, sna_final_answer = sna_results_map[i]

                    if sna_final_answer.lower() != "safe" and sna_final_answer != proposed_action and proposed_action!="stop":
                        detour_success = self._run_detour_task(
                            env=env,
                            model=model,
                            tokenizer=tokenizer,
                            agent_idx=i,
                            sna_instruction=sna_final_answer,
                            proposed_action=proposed_action,
                            ended_global=ended,
                            num_act_nav=num_act_nav,
                            trajs=trajs,
                            total_steps=total_steps
                        )

                        if detour_success == True:
                            action_to_execute = "<detour_skip>"
                        elif detour_success == "fail_detour":
                            action_to_execute = "stop"
                        else:
                            action_to_execute = proposed_action

                if action_to_execute == 'unk': action_to_execute = 'stop'
                final_actions.append(action_to_execute)

            env.env.action_step(final_actions, ended, num_act_nav, trajs, total_steps)

            if not num_act_nav[0]:
                break

        return trajs, [conv['obs'] for conv in conv_batch], responses


def inference(agent, env, dataloader, model, tokenizer):
    metrics = [0] * 5
    trajs_record = []
    for inputs in tqdm(dataloader):
        route_ids = inputs.pop('route_ids')
        trajs, images, responses = agent.rollout(env=env, model=model, tokenizer=tokenizer, route_ids=route_ids)
        if trajs is None: continue
        env.eva_metrics(trajs, metrics)
        for i in range(len(trajs)):
            trajs_record.append(
                {'route_ids': route_ids[i], 'navigation_text': env.dict_data[route_ids[i]]['navigation_text'],
                 'trajs': trajs[i], 'images': images[i], 'responses': responses[i]})

    eval_dataset = getattr(dataloader, "dataset", None)
    num_samples = len(eval_dataset)
    metrics = [value / num_samples for value in metrics]
    metrics = [
        metrics[0] * 100,
        metrics[1],
        metrics[2] * 100,
        metrics[3] * 100,
        metrics[4] * 100,
    ]
    return metrics, trajs_record


def run_navigation(eval_env, model, tokenizer, opts, sna_module=None):
    dataloader = make_evaluation_data_module(opts)
    split_name = eval_env.env.name
    eval_env.reset_epoch()

    agent = FLAME(opts, sna_module=sna_module)  # Initialize Agent

    metrics, trajs_record = inference(agent, eval_env, dataloader, model, tokenizer)
    metrics = {
        "TC": metrics[0],
        f"TC_{split_name}": metrics[0],
        f"SPD_{split_name}": metrics[1],
        f"SED_{split_name}": metrics[2],
        "SPL": metrics[4],
        f"SPL_{split_name}": metrics[4],
        "CVR": metrics[3],
        f"CVR_{split_name}": metrics[3]
    }
    return metrics, trajs_record
