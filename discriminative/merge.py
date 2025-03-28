import torch
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import torch.nn as nn
import copy
import sparsify
import utils
import eval
from collections import defaultdict, OrderedDict
from param import param
import random

class MergingMethod:

    @utils.args_inspector
    def __init__(
        self, 
        models_to_merge, 
        models_name,
    ):
        self.models_name = {n:i for i,n in enumerate(models_name)}
        # dict(zip(models_name, range(0, N)))
        self.models_to_merge = models_to_merge

    def get_model(self, model_name):
        return self.models_to_merge[self.models_name[model_name]]

    @utils.args_inspector
    # @torch.inference_mode()
    def average_merging(
        self, 
    ):

        merged_param = param.vectorize_reduce(
            lambda x: torch.stack(x).mean(dim=0), 
            self.models_to_merge
        )
        return merged_param

    @utils.args_inspector
    def fisher_merge(
        self, 
        models_to_merge: list, 
        data_names: list,
        data_nums: list, 
        fish_scaling: list = None,
        norm_fish_weight: bool = True, 
        min_fish_weight: float = 1e-6
    ):
        from merger.fisher_merge import FisherMerge
        merger = FisherMerge(
            models_to_merge, 
            data_names, data_nums, 
            fish_scaling, norm_fish_weight,min_fish_weight
        )
        return merger.merge()

    @utils.args_inspector
    # @torch.inference_mode()
    def regmean_merge(
        self,
        models_to_merge: list,
        data_names: list,
        data_nums: list, 
        reduce_non_diagonal_ratio: float = 1.0
    ):

        from merger.regmean_merge import RegMeanMerge
        merger = RegMeanMerge(
            models_to_merge, 
            data_names, data_nums, 
            reduce_non_diagonal_ratio,
        )
        return merger.merge()

    @utils.args_inspector
    # @torch.inference_mode()
    def ties_merge(
        self,
        base_model: nn.Module,
        models_to_merge: list,
        mask_rate: float = 0.8,
        scaling: float = 1.0,
    ):

        def disjoint_merge(
            tensor: torch.Tensor, # (n_model, n_para)
            merge_func:str = 'mean',
        ):

            sign = torch.sign(tensor.sum(dim=0)) # (num_total_params, )
            majority_sign = torch.sign(sign.sum(dim=0))
            # replace 0 in sign to the major sign in param_signs
            sign[sign == 0] = majority_sign
            del majority_sign

            # preserve the parameter with the expect sign
            mask = torch.where(
                sign.unsqueeze(0) > 0, tensor > 0, tensor < 0
            )
            tensor = tensor * mask
            
            # (n_model, n_para) -> (n_para,)
            if merge_func == "mean":
                num_ = (tensor != 0).sum(dim=0).float()
                # min=1.0 避免num_=0的情况
                tensor = torch.sum(tensor, dim=0) / torch.clamp(num_, min=1.0)
            elif merge_func == "sum":
                tensor = torch.sum(tensor, dim=0)
            elif merge_func == "max":
                tensor = tensor.abs().max(dim=0)[0]
                tensor *= sign
            return tensor

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        flattened_param = [ task_vector.flatten() for task_vector in task_vectors ]
        # sparsify on model-level => (n_model, n_para)
        # flattened_param = torch.vstack(
        #     [ sparsify.magnitude(_param, 1 - mask_rate) for _param in flattened_param ]
        # )

        def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
            if K == 100:
                # print("Not applying mask")
                if return_mask:
                    return M, torch.ones_like(M), None
                else:
                    return M, torch.ones_like(M)

            if K >= 1:
                K /= 100

            original_shape = M.shape
            if M.dim() == 1:
                M = M.unsqueeze(0)

            n, d = M.shape
            k = int(d * K)
            k = d - k  # Keep top k elements instead of bottom k elements

            # Find the k-th smallest element by magnitude for each row
            kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
            # Create a mask tensor with True for the top k elements in each row
            mask = M.abs() >= kth_values
            final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

            if reshape_mask:
                final_mask = final_mask.reshape(M.shape)

            if return_mask:
                return M * final_mask, final_mask.float().mean(dim=1), final_mask
            else:
                return M * final_mask, final_mask.float().mean(dim=1)
        
        # flattened_param1 = sparsify.magnitude(torch.vstack(flattened_param), 1 - mask_rate)
        flattened_param = topk_values_mask(torch.vstack(flattened_param), 1 - mask_rate)[0]
        flattened_param = disjoint_merge(flattened_param)
        # randomly select one vector to unflatten
        merged_param = copy.deepcopy(base_model)
        merged_param = base_model + scaling * merged_param.unflatten(flattened_param)
        return merged_param

    @utils.args_inspector
    # @torch.inference_mode()
    def ties_merge_old(
        self,
        base_model: nn.Module,
        models_to_merge: list,
        mask_rate: float = 0.8,
        scaling: float = 1.0,
    ):

        def state_dict_to_vector(state_dict, remove_keys=[]):
            shared_state_dict = copy.deepcopy(state_dict)
            for key in remove_keys:
                if key in shared_state_dict:
                    del shared_state_dict[key]
            sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
            return torch.nn.utils.parameters_to_vector([value.reshape(-1) for key, value in sorted_shared_state_dict.items()])


        def vector_to_state_dict(vector, state_dict, remove_keys=[]):
            # create a reference dict to define the order of the vector
            reference_dict = copy.deepcopy(state_dict)
            for key in remove_keys:
                if key in reference_dict:
                    del reference_dict[key]
            sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

            # create a shared state dict using the refence dict
            torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

            # add back the encoder and decoder embedding weights.
            if "transformer.shared.weight" in sorted_reference_dict:
                for key in remove_keys:
                    sorted_reference_dict[key] = sorted_reference_dict["transformer.shared.weight"]
            return sorted_reference_dict

        def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
            if K == 100:
                # print("Not applying mask")
                if return_mask:
                    return M, torch.ones_like(M), None
                else:
                    return M, torch.ones_like(M)

            if K >= 1:
                K /= 100

            original_shape = M.shape
            if M.dim() == 1:
                M = M.unsqueeze(0)

            n, d = M.shape
            k = int(d * K)
            k = d - k  # Keep top k elements instead of bottom k elements

            # Find the k-th smallest element by magnitude for each row
            kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
            # Create a mask tensor with True for the top k elements in each row
            mask = M.abs() >= kth_values
            final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

            if reshape_mask:
                final_mask = final_mask.reshape(M.shape)

            if return_mask:
                return M * final_mask, final_mask.float().mean(dim=1), final_mask
            else:
                return M * final_mask, final_mask.float().mean(dim=1)

        def resolve_sign(tensor: torch.Tensor):
            sign_to_mult = torch.sign(tensor.sum(dim=0))
            sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
            return sign_to_mult

        def resolve_zero_signs(sign_to_mult, method="majority"):
            majority_sign = torch.sign(sign_to_mult.sum())

            if method == "majority":
                sign_to_mult[sign_to_mult == 0] = majority_sign
            elif method == "minority":
                sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
            return sign_to_mult

        def disjoint_merge(tensor, merge_func, sign_to_mult):
            merge_func = merge_func.split("-")[-1]

            # If sign is provided then we select the corresponding entries and aggregate.
            if sign_to_mult is not None:
                rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, tensor > 0, tensor < 0)
                selected_entries = tensor * rows_to_keep
            # Else we select all non-zero entries and aggregate.
            else:
                rows_to_keep = tensor != 0
                selected_entries = tensor * rows_to_keep

            if merge_func == "mean":
                non_zero_counts = (selected_entries != 0).sum(dim=0).float()
                disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
            elif merge_func == "sum":
                disjoint_aggs = torch.sum(selected_entries, dim=0)
            elif merge_func == "max":
                disjoint_aggs = selected_entries.abs().max(dim=0)[0]
                disjoint_aggs *= sign_to_mult
            else:
                raise ValueError(f"Merge method {merge_func} is not defined.")

            return disjoint_aggs

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        flattened_param = [state_dict_to_vector(task_vector.param_dict) for task_vector in task_vectors ]
        all_checks = torch.vstack(flattened_param)
        updated_checks, *_ = topk_values_mask(all_checks, K=1 - mask_rate, return_mask=False)
        print(f"RESOLVING SIGN")
        final_signs = resolve_sign(updated_checks)
        assert final_signs is not None

        print(f"Disjoint AGGREGATION: dis-mean")
        merged_tv = disjoint_merge(updated_checks, 'dis-mean', final_signs)
        merged_tv_state_dict = vector_to_state_dict(merged_tv, copy.deepcopy(base_model.param_dict))
        merged_param = base_model + scaling * param(merged_tv_state_dict)
        return merged_param


    @utils.args_inspector
    # @torch.inference_mode()
    def task_arithmetic(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        scaling: float = 1.0,
    ):

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        merged_param = base_model + scaling * sum(task_vectors)
        return merged_param

    @utils.args_inspector
    # @torch.inference_mode()
    def task_arithmetic_search(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        scaling: float = 1.0,
    ):

        task_vectors = [
            model - base_model
            for model in models_to_merge
        ]
        
        merged_param = base_model + sum([
            w * tv
            for w, tv in zip(scaling, task_vectors)
        ])
        return merged_param

    @utils.args_inspector
    # @torch.inference_mode()
    def task_arithmetic_plus(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        scaling: float = 1.0,
        mask_strategy: str = None, 
        mask_rate: float = None,
    ):

        task_vectors = [
            model + base_model
            for model in models_to_merge
        ]
        
        if mask_strategy is None:
            merged_param = (scaling * sum(task_vectors)) - base_model
        else: 
            merged_param = (scaling * sum(task_vectors)).map(
            lambda n,p: getattr(sparsify, mask_strategy)(
                p, 
                1 - mask_rate,
            ),
            desc=mask_strategy
        )- base_model
        return merged_param

    @utils.args_inspector
    # @torch.inference_mode()
    def dare_merge(
        self, 
        models_to_merge: param,
        second_merge_method: str,
        second_merge_config: dict,
        mask_rate: float,
        base_model: nn.Module,
        mask_scale: float = 1.0,
        weight_format: str = 'delta',
    ):
        # 1. sparsify masking (merge with base model)
        masked_params = [
            self.dare_mask(
                finetuned_model,
                mask_rate,
                base_model,
                mask_scale,
                weight_format,
            ) for finetuned_model in models_to_merge
        ]
        # 2. merge between the different models
        merged_params = getattr(self, second_merge_method)(
            base_model = base_model,
            models_to_merge = masked_params,
            **second_merge_config
        )
        return merged_params

    # @torch.inference_mode()
    def dare_mask(
        self,
        finetuned_model: nn.Module, 
        mask_rate: float, 
        base_model: nn.Module = None, 
        mask_scale: float = 1.0,
        weight_format: str = 'delta'
    ):

        mask_rate = float(mask_rate)

        if weight_format == "full" or weight_format == "lora":
            masked_param = finetuned_model
        elif weight_format == "delta":
            masked_param = finetuned_model - base_model
        else:
            raise NotImplementedError

        def mask_input_with_mask_rate(input_tensor: torch.Tensor, density: float, use_rescale: bool = True, mask_strategy: str = 'random'):
            mask_rate = 1 - density
            assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
            if mask_strategy == "random":
                mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
                masked_input_tensor = input_tensor * (1 - mask)
            else:
                assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
                original_shape = input_tensor.shape
                input_tensor = input_tensor.flatten()
                num_mask_params = int(len(input_tensor) * mask_rate)
                # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
                kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
                # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
                mask = input_tensor.abs() <= kth_values
                masked_input_tensor = input_tensor * (~mask)
                masked_input_tensor = masked_input_tensor.reshape(original_shape)
            if use_rescale and mask_rate != 1.0:
                masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
            return masked_input_tensor
        
        # mask_input_with_mask_rate
        masked_param = masked_param.map(
            lambda n,p: sparsify.bernoulli(
                p, 
                1 - mask_rate,
            ),
            desc='bernoulli'
        )
        
        if weight_format == "delta":
            masked_param = base_model + mask_scale * masked_param
        return masked_param

    @utils.args_inspector
    # @torch.inference_mode()
    def twin_merge(
        self,
        base_model: nn.Module,
        models_to_merge: param,
        second_merge_method: str,
        second_merge_config: dict,
    ):
        # merge again / MergePlus / DoubleBundle / DualMerger

        # Get merged parameter
        merged_params = getattr(self, second_merge_method)(
            base_model = base_model,
            models_to_merge = models_to_merge,
            **second_merge_config
        )
        return merged_params

    from datasets import load_dataset
    @utils.args_inspector
    # @torch.inference_mode()
    def frank_wolfe_merge(
        self,
        base_model: nn.Module,
        models_finetuned: dict,
        models_to_merge: list,
        proxy_dataset_path: str,
        second_merge_method: str,
        second_merge_config: dict,
        max_iters: int = 5,
        step_size: float = 1,
        reg_factor: float = 1.0,
        eval_tasks: list = set(),
        src_merge: list = [],
    ):
        """
        Merges models using the Frank-Wolfe optimization algorithm.

        Args:
            base_model (nn.Module): The base model to start from.
            models_to_merge (list): List of models to merge.
            proxy_dataset_path (str): Path to the proxy dataset.
            second_merge_method (str): Method to merge the models.
            second_merge_config (dict): Configuration for the second merge method.
            max_iters (int): Maximum number of iterations.
            step_size (float): Step size for the optimization.
            reg_factor (float): Regularization factor for regression
            eval_tasks (list): List of tasks to evaluate on.
            src_merge (list): List of source models to merge.

        Returns:
        nn.Module: Merged model.
        """
        # Function to compute gradient of w^t
        def compute_gradient(models_finetuned: dict, merged_model: dict, train_dataset, eval_tasks):
            gradients = {}

            # Ensure model parameters require gradients and reset gradients
            for param in merged_model.values():
                param.requires_grad = True
                param.grad = None

            # Zero the gradients of models
            for model in models_finetuned.values():
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                    param.grad = None

            avg_loss = defaultdict(float)
            # Compute gradients outside inference mode
            for data_item in train_dataset:
                data_id = data_item['dataset_ids']
                data_name = list(eval.glue_data_id_map.keys())[data_id]
                if data_name not in eval_tasks:
                    continue
                model = models_finetuned[data_name]

                # Prepare inputs (no gradients required for these)
                input_ids = torch.tensor(data_item['input_ids']).unsqueeze(0).to(model.device)
                attention_mask = torch.tensor(data_item['attention_mask']).unsqueeze(0).to(model.device)
                label = torch.tensor(data_item['label']).unsqueeze(0).to(model.device)

                # Compute the output and loss
                output = torch.func.functional_call(
                        model,
                        merged_model.param_dict,  # Pass the merged model parameters
                        args=(input_ids, attention_mask),
                ).logits

                if label.dtype == torch.long:
                    loss = torch.nn.functional.cross_entropy(output, label)
                else:
                    loss = torch.nn.functional.mse_loss(output, label) * reg_factor  # Scale down the loss for regression tasks

                # Backpropagate the gradients
                loss.backward()
                avg_loss[data_name] += loss.item()
            
            avg_loss = {k: v / len(train_dataset) for k, v in avg_loss.items()}
            print(f"Average Loss: {avg_loss}, Total Loss: {sum(avg_loss.values()) / len(avg_loss)}")

            # Save the gradients
            for name, param in merged_model.items():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()

            return gradients

        def frank_wolfe_selection(gradients, models_to_merge, src_merge, granularity='task'):
            if granularity == 'layer':
                # Find the task vector with the most alignment to the gradient
                model_to_merge_dict = {}
                min_alignment = {}
                min_idx = {}
                for i, model_to_merge in enumerate(models_to_merge):
                    for param_name, param_value in model_to_merge.items():
                        # caclulate consine similarity
                        grad = gradients[param_name]
                        ckpt = model_to_merge[param_name]
                        param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / (torch.norm(grad) * torch.norm(ckpt))
                        if param_name not in min_alignment or param_alignment < min_alignment[param_name]:
                            min_alignment[param_name] = param_alignment
                            if min_alignment[param_name] < 0:
                                model_to_merge_dict[param_name] = param_value
                                min_idx[param_name] = i
                            else:
                                model_to_merge_dict[param_name] = torch.zeros_like(param_value)
                model_to_merge = param(model_to_merge_dict)
                chosen_model = {model_name: 0 for model_name in src_merge}
                for k in min_idx.values():
                    chosen_model[src_merge[k]] += 1
                return model_to_merge, chosen_model
            else:
                min_inner_product = float('inf')
                min_model = None
                min_model_name = None
                for model_name, model_to_merge in zip(src_merge, models_to_merge):
                    inner_product_sum = 0
                    for param_name, param_value in model_to_merge.items():
                        # caclulate consine similarity
                        grad = gradients[param_name]
                        ckpt = model_to_merge[param_name]
                        param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / (torch.norm(grad) * torch.norm(ckpt))
                        inner_product_sum += param_alignment
                    if inner_product_sum < min_inner_product:
                        min_inner_product = inner_product_sum
                        min_model = model_to_merge
                        min_model_name = model_name
                return min_model, min_model_name

        # Initialize the merged model as the base model
        # merged_model = copy.deepcopy(base_model)
        merged_model = getattr(self, second_merge_method)(
                base_model=copy.deepcopy(base_model),
                models_to_merge=models_to_merge,
                **second_merge_config
            )
        dataset = utils.from_json(proxy_dataset_path)
        scaling = second_merge_config['scaling'] if 'scaling' in second_merge_config else 1.0
        batch_size = len(dataset)

        models_to_merge.append(copy.deepcopy(param(merged_model)))
        src_merge.append('initial')
        for iteration in tqdm(range(max_iters), desc="Frank-Wolfe Iteration"):
            torch.cuda.empty_cache()
        
            torch.set_grad_enabled(True)
            # Compute the gradient based on a batch of data, dataset is a list of dictionaries
            dataset_batch = random.sample(dataset, batch_size)
            gradients = compute_gradient(models_finetuned, merged_model, dataset_batch, eval_tasks)
            torch.set_grad_enabled(False)
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients.values()]))

            model_to_merge, chosen_model = frank_wolfe_selection(gradients, models_to_merge, src_merge, granularity='layer')
            
            # Determine step size
            step = 2 / (((batch_size * iteration) // len(dataset) ) + 2) * step_size
            
            # print iteration information
            print(f"Iteration {iteration+1}, Task Vector: {chosen_model}, Gradient Norm: {grad_norm:.6f}, Step Size: {step:.6f}")
            
            second_merge_config['scaling'] = scaling * step
            # Update merged model parameters using second merge method
            merged_model = getattr(self, second_merge_method)(
                base_model=merged_model,
                models_to_merge=[model_to_merge],
                **second_merge_config
            )
            
            for model in models_finetuned.values():
                model.eval()
            del model_to_merge, chosen_model

        return merged_model