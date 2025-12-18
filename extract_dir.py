import argparse
import os
import torch
from hook_utils import InterventionDirection, InterventionDirectionComponent


def build_direction(label: str,
                    positive_activations: dict,
                    reference_activations: dict,
                    all_activations: dict) -> InterventionDirection:
    direction = InterventionDirection(components={})
    for component, pos_tensor in positive_activations.items():
        if component not in reference_activations or component not in all_activations:
            continue
        ref_tensor = reference_activations[component]
        all_tensor = all_activations[component]
        if pos_tensor.numel() == 0:
            print(f"[{label}] component {component}: no positive activations, skipping.")
            continue
        if ref_tensor.numel() == 0:
            print(f"[{label}] component {component}: no reference activations, skipping.")
            continue
        component_dir = InterventionDirectionComponent()
        mean_pos = pos_tensor.mean(dim=0)
        mean_ref = ref_tensor.mean(dim=0)
        diff = mean_pos - mean_ref
        norm = diff.norm()
        norm_value = norm.item()
        if norm_value == 0 or torch.isnan(norm).item():
            print(f"[{label}] component {component}: zero/NaN norm, skipping.")
            continue
        component_dir.mean_diff = diff
        normed_mean_diff = diff / norm
        component_dir.mean_pos = (mean_pos.T @ normed_mean_diff).item()
        component_dir.std_pos = (pos_tensor @ normed_mean_diff).std()
        component_dir.mean_neg = (mean_ref.T @ normed_mean_diff).item()
        component_dir.std_neg = (ref_tensor @ normed_mean_diff).std()
        component_dir.mean_all = (all_tensor.mean(dim=0).T @ normed_mean_diff).item()
        component_dir.pos_ratio = pos_tensor.shape[0] / max(ref_tensor.shape[0], 1)
        direction.components[component] = component_dir
        print(f"[{label}] component: {component}")
        print(f"[{label}] diff norm: {norm_value}")
        print(f"[{label}] mean_pos: {component_dir.mean_pos}, std_pos: {component_dir.std_pos}, "
              f"mean_neg: {component_dir.mean_neg}, std_neg: {component_dir.std_neg}")
    return direction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-r1-qwen-1.5b")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--instruction", type=str, default="")
    args = parser.parse_args()

    data = torch.load(f"activations/{args.model}/{args.dataset}/{args.instruction}/activations.pt")

    activations = data['activation_stores']
    reflect_mask = torch.tensor(data['is_reflect_stores'], dtype=torch.bool)
    end_mask = torch.tensor(data['is_end_stores'], dtype=torch.bool)
    num_steps = reflect_mask.shape[0]

    def get_optional_mask(name: str) -> torch.Tensor:
        values = data.get(name)
        if values is None:
            return torch.zeros(num_steps, dtype=torch.bool)
        return torch.tensor(values, dtype=torch.bool)

    conclusion_mask = get_optional_mask('is_conclusion_stores')
    post_conclusion_mask = get_optional_mask('is_post_conclusion_stores')

    reflect_pre_mask_values = data.get('is_reflect_pre_conclusion_stores')
    if reflect_pre_mask_values is not None:
        reflect_pre_mask = torch.tensor(reflect_pre_mask_values, dtype=torch.bool)
    else:
        reflect_pre_mask = reflect_mask & (~post_conclusion_mask) & (~conclusion_mask)

    reflect_post_mask_values = data.get('is_reflect_post_conclusion_stores')
    if reflect_post_mask_values is not None:
        reflect_post_mask = torch.tensor(reflect_post_mask_values, dtype=torch.bool)
    else:
        reflect_post_mask = reflect_mask & post_conclusion_mask

    remaining_mask = (~reflect_mask) & (~end_mask)
    remaining_pre_mask = remaining_mask & (~post_conclusion_mask) & (~conclusion_mask)
    remaining_post_mask = remaining_mask & post_conclusion_mask

    print(f"reflect num: {reflect_mask.sum().item()}/{num_steps}")
    print(f"end num: {end_mask.sum().item()}/{num_steps}")
    print(f"pre-conclusion reflect num: {reflect_pre_mask.sum().item()}/{num_steps}")
    print(f"post-conclusion reflect num: {reflect_post_mask.sum().item()}/{num_steps}")

    activations_reflect, activations_end = {}, {}
    activations_remaining, activations_remaining_pre, activations_remaining_post = {}, {}, {}
    activations_reflect_pre, activations_reflect_post = {}, {}
    for component, tensor in activations.items():
        activations_reflect[component] = tensor[reflect_mask]
        activations_end[component] = tensor[end_mask]
        activations_remaining[component] = tensor[remaining_mask]
        activations_reflect_pre[component] = tensor[reflect_pre_mask]
        activations_reflect_post[component] = tensor[reflect_post_mask]
        activations_remaining_pre[component] = tensor[remaining_pre_mask]
        activations_remaining_post[component] = tensor[remaining_post_mask]

    reflect_dir = build_direction("reflect", activations_reflect, activations_remaining, activations)
    endthink_dir = build_direction("end-think", activations_end, activations_remaining, activations)
    pre_conclu_reflect_dir = build_direction(
        "pre-conclusion reflect", activations_reflect_pre, activations_remaining_pre, activations
    )
    post_conclu_reflect_dir = build_direction(
        "post-conclusion reflect", activations_reflect_post, activations_remaining_post, activations
    )

    save_dir = f"intervention_direction/{args.model}/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    reflect_dir.save(os.path.join(save_dir, "reflect_dir.pt"))
    endthink_dir.save(os.path.join(save_dir, "endthink_dir.pt"))
    pre_conclu_reflect_dir.save(os.path.join(save_dir, "pre_conclu_reflect_dir.pt"))
    post_conclu_reflect_dir.save(os.path.join(save_dir, "post_conclu_reflect_dir.pt"))
    print(f"Intervention direction saved to {save_dir}")
