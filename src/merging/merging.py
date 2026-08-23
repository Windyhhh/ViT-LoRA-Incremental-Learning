import torch
from configs.config import Config

def merge_loras_sd_lora_inspired(merged_lora_state, new_lora_state, task_id,
                                  magnitude_history=None, direction_history=None):
    """
    SD-LoRA inspired merge: Decouple magnitude and direction

    Key idea from arXiv:2501.13198:
    - Fix directions learned from previous tasks
    - Only learn magnitudes (alphas) for each direction
    - Output: W = W_0 + Σ α_k · (A_k B_k / ||A_k B_k||)

    Args:
        merged_lora_state: Previously merged LoRA state (or None for first task)
        new_lora_state: New task's LoRA state
        task_id: Current task ID (1-indexed)
        magnitude_history: List of learned magnitudes [α_1, α_2, ...]
        direction_history: List of normalized directions [D_1, D_2, ...]

    Returns:
        merged_state: New merged LoRA state
        magnitude_history: Updated magnitude history
        direction_history: Updated direction history
    """
    if magnitude_history is None:
        magnitude_history = []
    if direction_history is None:
        direction_history = []

    # First task: just normalize and store
    if task_id == 1:
        merged_state = {}
        layers = set([k.rsplit('.', 1)[0] for k in new_lora_state.keys()])

        for layer_name in layers:
            A = new_lora_state[f'{layer_name}.lora_A']
            B = new_lora_state[f'{layer_name}.lora_B']

            # Compute ΔW = B @ A
            delta_W = B @ A

            # Compute magnitude and direction
            magnitude = torch.norm(delta_W, p='fro')
            direction = delta_W / (magnitude + 1e-8)

            # Store direction
            if len(direction_history) == 0:
                direction_history.append({})
            direction_history[0][layer_name] = direction

            # Store magnitude (α_1 = 1.0 for first task)
            if len(magnitude_history) == 0:
                magnitude_history.append(1.0)

            # Reconstruct LoRA: ΔW = α * direction
            scaled_delta_W = magnitude_history[0] * direction_history[0][layer_name]

            # Decompose back to A, B using SVD
            U, S, Vh = torch.linalg.svd(scaled_delta_W, full_matrices=False)
            rank = min(Config.MIN_MERGED_RANK, len(S))

            sqrt_S = torch.sqrt(S[:rank])
            merged_state[f'{layer_name}.lora_A'] = torch.diag(sqrt_S) @ Vh[:rank, :]
            merged_state[f'{layer_name}.lora_B'] = U[:, :rank] @ torch.diag(sqrt_S)

        return merged_state, magnitude_history, direction_history

    # Subsequent tasks: add new direction with learned magnitude
    merged_state = {}
    layers = set([k.rsplit('.', 1)[0] for k in new_lora_state.keys()])

    # Add new direction from current task
    direction_history.append({})

    for layer_name in layers:
        A_new = new_lora_state[f'{layer_name}.lora_A']
        B_new = new_lora_state[f'{layer_name}.lora_B']

        # Compute new task's ΔW
        delta_W_new = B_new @ A_new

        # Normalize to get direction
        magnitude_new = torch.norm(delta_W_new, p='fro')
        direction_new = delta_W_new / (magnitude_new + 1e-8)

        # Store new direction
        direction_history[-1][layer_name] = direction_new

        # Compute magnitude for new task (adaptive based on previous magnitudes)
        # α_t = average of previous magnitudes (conservative approach)
        avg_magnitude = sum(magnitude_history) / len(magnitude_history)
        new_magnitude = 0.8 * avg_magnitude  # Slightly reduce to prevent drift

        # Reconstruct merged ΔW = Σ α_k · D_k
        delta_W_merged = torch.zeros_like(direction_new)
        for k, (alpha, directions) in enumerate(zip(magnitude_history, direction_history[:-1])):
            delta_W_merged += alpha * directions[layer_name]

        # Add new task's contribution
        delta_W_merged += new_magnitude * direction_new

        # Decompose back to A, B
        U, S, Vh = torch.linalg.svd(delta_W_merged, full_matrices=False)
        rank = min(Config.MIN_MERGED_RANK, len(S))

        sqrt_S = torch.sqrt(S[:rank])
        merged_state[f'{layer_name}.lora_A'] = torch.diag(sqrt_S) @ Vh[:rank, :]
        merged_state[f'{layer_name}.lora_B'] = U[:, :rank] @ torch.diag(sqrt_S)

    # Update magnitude history
    magnitude_history.append(new_magnitude)

    return merged_state, magnitude_history, direction_history

def merge_loras_orthogonal_projection(merged_lora_state, new_lora_state, task_id,
                                       projection_threshold=0.5):
    """
    Orthogonal Projection-based Continual Merging (OPCM)

    Key idea from arXiv:2501.09522:
    - Project new task vector onto subspace orthogonal to merged task vector
    - Use adaptive scaling to maintain stable parameter distance
    - Minimize interference between tasks

    Args:
        merged_lora_state: Previously merged LoRA state (or None for first task)
        new_lora_state: New task's LoRA state
        task_id: Current task ID (1-indexed)
        projection_threshold: α in paper (controls orthogonal projection)

    Returns:
        merged_state: New merged LoRA state
    """
    # First task: just return as-is
    if task_id == 1 or merged_lora_state is None:
        return new_lora_state

    merged_state = {}
    layers = set([k.rsplit('.', 1)[0] for k in new_lora_state.keys()])

    for layer_name in layers:
        # Get merged and new LoRA parameters
        A_merged = merged_lora_state[f'{layer_name}.lora_A']
        B_merged = merged_lora_state[f'{layer_name}.lora_B']
        A_new = new_lora_state[f'{layer_name}.lora_A']
        B_new = new_lora_state[f'{layer_name}.lora_B']

        # Compute task vectors: ΔW = B @ A
        delta_W_merged = B_merged @ A_merged
        delta_W_new = B_new @ A_new

        # SVD of merged task vector
        U, S, Vh = torch.linalg.svd(delta_W_merged, full_matrices=False)

        # Determine rank threshold based on projection_threshold
        total_energy = S.sum()
        cumsum_energy = torch.cumsum(S, dim=0)
        r_alpha = torch.searchsorted(cumsum_energy, projection_threshold * total_energy).item() + 1
        r_alpha = min(r_alpha, len(S))

        # Orthogonal projection: P_α(ΔW_new)
        # Project onto subspace spanned by {u_i v_j^T} where i≠j or i,j >= r_α
        delta_W_proj = torch.zeros_like(delta_W_new)

        for i in range(delta_W_new.shape[0]):
            for j in range(delta_W_new.shape[1]):
                # Skip if within the principal subspace
                if i < r_alpha and j < r_alpha and i == j:
                    continue

                # Compute projection coefficient
                if i < len(U) and j < len(Vh):
                    u_i = U[:, i:i+1]
                    v_j = Vh[j:j+1, :]
                    basis = u_i @ v_j

                    # <ΔW_new, u_i v_j^T>_F
                    coeff = torch.sum(delta_W_new * basis)
                    delta_W_proj += coeff * basis

        # Adaptive scaling: λ^(t) = ||λ^(t-1) ΔW_merged + P_α(ΔW_new)|| / avg_norm
        # For simplicity, use sqrt(t) scaling as suggested in paper
        lambda_t = torch.sqrt(torch.tensor(task_id, dtype=torch.float32))

        # Merge: ΔW_merged^(t) = (ΔW_merged + P_α(ΔW_new)) / λ^(t)
        delta_W_merged_new = (delta_W_merged + delta_W_proj) / lambda_t

        # Decompose back to A, B
        U_new, S_new, Vh_new = torch.linalg.svd(delta_W_merged_new, full_matrices=False)
        rank = min(Config.MIN_MERGED_RANK, len(S_new))

        sqrt_S_new = torch.sqrt(S_new[:rank])
        merged_state[f'{layer_name}.lora_A'] = torch.diag(sqrt_S_new) @ Vh_new[:rank, :]
        merged_state[f'{layer_name}.lora_B'] = U_new[:, :rank] @ torch.diag(sqrt_S_new)

    return merged_state
