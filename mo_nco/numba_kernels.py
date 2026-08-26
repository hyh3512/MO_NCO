from __future__ import annotations

try:
    import numpy as np
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional accelerator fallback
    np = None  # type: ignore[assignment]
    njit = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def scalar_two_opt_descent_numba(matrix, tour, max_passes):  # type: ignore[no-untyped-def]
        current = tour.copy()
        n = current.shape[0]
        for _ in range(max_passes):
            best_delta = -1e-12
            best_i = -1
            best_j = -1
            for i in range(1, n - 1):
                a = current[i - 1]
                b = current[i]
                for j in range(i + 1, n):
                    c = current[j]
                    d = current[(j + 1) % n]
                    delta = matrix[a, c] + matrix[b, d] - matrix[a, b] - matrix[c, d]
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_i < 0:
                break
            left = best_i
            right = best_j
            while left < right:
                tmp = current[left]
                current[left] = current[right]
                current[right] = tmp
                left += 1
                right -= 1
        return current

    @njit(cache=True)
    def scalar_relocate_descent_numba(matrix, tour, max_passes):  # type: ignore[no-untyped-def]
        current = tour.copy()
        n = current.shape[0]
        for _ in range(max_passes):
            best_delta = -1e-12
            best_i = -1
            best_j = -1
            for i in range(1, n):
                a = current[i - 1]
                v = current[i]
                b = current[(i + 1) % n]
                remove_delta = matrix[a, b] - matrix[a, v] - matrix[v, b]
                for j in range(n):
                    if j == i or j == i - 1:
                        continue
                    c = current[j]
                    d = current[(j + 1) % n]
                    delta = remove_delta + matrix[c, v] + matrix[v, d] - matrix[c, d]
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            if best_i < 0:
                break
            city = current[best_i]
            insert_at = best_j + 1 if best_j < best_i else best_j
            if best_i < insert_at:
                for k in range(best_i, insert_at):
                    current[k] = current[k + 1]
                current[insert_at] = city
            elif best_i > insert_at:
                for k in range(best_i, insert_at, -1):
                    current[k] = current[k - 1]
                current[insert_at] = city
        return current

    @njit(cache=True)
    def _cycle_cost2(matrix, tour):  # type: ignore[no-untyped-def]
        total = 0.0
        n = tour.shape[0]
        for i in range(n):
            total += matrix[tour[i], tour[(i + 1) % n]]
        return total

    @njit(cache=True)
    def _base_scalar2_numba(objective0, objective1, weight0, weight1, ideal0, ideal1, inv0, inv1):  # type: ignore[no-untyped-def]
        z0 = (objective0 - ideal0) * inv0
        z1 = (objective1 - ideal1) * inv1
        a = weight0 * z0
        b = weight1 * z1
        return (a if a >= b else b) + 0.03 * (a + b)

    @njit(cache=True)
    def two_opt_objectives_batch_numba(matrices, tour, objective0, objective1, pairs):  # type: ignore[no-untyped-def]
        count = pairs.shape[0]
        out = np.empty((count, 2), dtype=np.float64)
        n = tour.shape[0]
        for idx in range(count):
            i = pairs[idx, 0]
            j = pairs[idx, 1]
            if i > j:
                tmp = i
                i = j
                j = tmp
            if i <= 0 or j >= n or j - i <= 1:
                out[idx, 0] = objective0
                out[idx, 1] = objective1
                continue
            a = tour[i - 1]
            b = tour[i]
            c = tour[j]
            d = tour[(j + 1) % n]
            out[idx, 0] = objective0 - matrices[0, a, b] - matrices[0, c, d] + matrices[0, a, c] + matrices[0, b, d]
            out[idx, 1] = objective1 - matrices[1, a, b] - matrices[1, c, d] + matrices[1, a, c] + matrices[1, b, d]
        return out

    @njit(cache=True)
    def _rotate_to_zero_numba(tour):  # type: ignore[no-untyped-def]
        n = tour.shape[0]
        zero_pos = 0
        for i in range(n):
            if tour[i] == 0:
                zero_pos = i
                break
        if zero_pos == 0:
            return tour
        rotated = np.empty(n, dtype=np.int64)
        for i in range(n):
            rotated[i] = tour[(zero_pos + i) % n]
        return rotated

    @njit(cache=True)
    def _scalar_greedy_from_start_numba(  # type: ignore[no-untyped-def]
        matrices,
        start,
        weight0,
        weight1,
        scale0,
        scale1,
        greedy_candidate_pool,
    ):
        n = matrices.shape[1]
        tour = np.empty(n, dtype=np.int64)
        unvisited = np.ones(n, dtype=np.bool_)
        tour[0] = start
        unvisited[start] = False
        current = start
        top_pool = max(1, greedy_candidate_pool)
        top_idx = np.empty(top_pool, dtype=np.int64)
        top_score = np.empty(top_pool, dtype=np.float64)
        for pos in range(1, n):
            top_count = 0
            for city in range(n):
                if not unvisited[city]:
                    continue
                score = (
                    weight0 * matrices[0, current, city] / scale0
                    + weight1 * matrices[1, current, city] / scale1
                )
                if top_count < top_pool:
                    top_idx[top_count] = city
                    top_score[top_count] = score
                    top_count += 1
                    k = top_count - 1
                    while k > 0 and top_score[k] < top_score[k - 1]:
                        tmp_score = top_score[k - 1]
                        tmp_idx = top_idx[k - 1]
                        top_score[k - 1] = top_score[k]
                        top_idx[k - 1] = top_idx[k]
                        top_score[k] = tmp_score
                        top_idx[k] = tmp_idx
                        k -= 1
                elif score < top_score[top_pool - 1]:
                    top_score[top_pool - 1] = score
                    top_idx[top_pool - 1] = city
                    k = top_pool - 1
                    while k > 0 and top_score[k] < top_score[k - 1]:
                        tmp_score = top_score[k - 1]
                        tmp_idx = top_idx[k - 1]
                        top_score[k - 1] = top_score[k]
                        top_idx[k - 1] = top_idx[k]
                        top_score[k] = tmp_score
                        top_idx[k] = tmp_idx
                        k -= 1
            choice = np.random.randint(0, top_count)
            current = top_idx[choice]
            tour[pos] = current
            unvisited[current] = False
        return _rotate_to_zero_numba(tour)

    @njit(cache=True)
    def scalar_greedy_population_numba(  # type: ignore[no-untyped-def]
        matrices,
        weighted_matrices,
        weight0,
        weight1,
        edge_scale0,
        edge_scale1,
        population_size,
        greedy_start_pool,
        greedy_candidate_pool,
        seed,
        two_opt_passes,
        relocate_passes,
    ):
        np.random.seed(seed)
        n = matrices.shape[1]
        pop = np.empty((population_size, n), dtype=np.int64)
        objs = np.empty((population_size, 2), dtype=np.float64)
        for idx in range(population_size):
            best_tour = np.empty(n, dtype=np.int64)
            best_score = 1e300
            starts = max(1, greedy_start_pool)
            for start_pos in range(starts):
                if start_pos == 0 or n <= 1:
                    start = 0
                else:
                    start = np.random.randint(1, n)
                candidate = _scalar_greedy_from_start_numba(
                    matrices,
                    start,
                    weight0[idx],
                    weight1[idx],
                    edge_scale0,
                    edge_scale1,
                    greedy_candidate_pool,
                )
                score = _cycle_cost2(weighted_matrices[idx], candidate)
                if score < best_score:
                    best_score = score
                    best_tour = candidate
            if two_opt_passes > 0:
                best_tour = scalar_two_opt_descent_numba(weighted_matrices[idx], best_tour, two_opt_passes)
            if relocate_passes > 0:
                best_tour = scalar_relocate_descent_numba(weighted_matrices[idx], best_tour, relocate_passes)
            pop[idx, :] = best_tour
            objs[idx, 0] = _cycle_cost2(matrices[0], best_tour)
            objs[idx, 1] = _cycle_cost2(matrices[1], best_tour)
        return pop, objs

    @njit(cache=True)
    def ips_scalar_polish_epoch_numba(  # type: ignore[no-untyped-def]
        matrices,
        weighted_matrices,
        population,
        objectives,
        weight0,
        weight1,
        neighbors,
        max_children,
        start_step,
        seed,
        two_opt_passes,
        relocate_passes,
        ideal0,
        ideal1,
        nadir0,
        nadir1,
        current_rejection_streak,
        max_rejection_streak,
    ):
        np.random.seed(seed)
        pop = population.copy()
        objs = objectives.copy()
        num_particles = pop.shape[0]
        num_cities = pop.shape[1]
        neighbor_count = neighbors.shape[1]
        child_tours = np.empty((max_children, num_cities), dtype=np.int64)
        child_objs = np.empty((max_children, 2), dtype=np.float64)
        attempts = 0
        accepted = 0
        rejected = 0
        inv0 = 1.0 / max(1e-9, nadir0 - ideal0)
        inv1 = 1.0 / max(1e-9, nadir1 - ideal1)

        for child_idx in range(max_children):
            step = start_step + child_idx
            direction_idx = step % num_particles
            best_parent = neighbors[direction_idx, 0]
            best_value = _base_scalar2_numba(
                objs[best_parent, 0],
                objs[best_parent, 1],
                weight0[direction_idx],
                weight1[direction_idx],
                ideal0,
                ideal1,
                inv0,
                inv1,
            )
            for pos in range(1, neighbor_count):
                candidate_idx = neighbors[direction_idx, pos]
                value = _base_scalar2_numba(
                    objs[candidate_idx, 0],
                    objs[candidate_idx, 1],
                    weight0[direction_idx],
                    weight1[direction_idx],
                    ideal0,
                    ideal1,
                    inv0,
                    inv1,
                )
                if value < best_value:
                    best_value = value
                    best_parent = candidate_idx

            child = pop[best_parent].copy()
            i = np.random.randint(1, num_cities)
            j = np.random.randint(1, num_cities - 1)
            if j >= i:
                j += 1
            if i > j:
                tmp = i
                i = j
                j = tmp
            left = i
            right = j
            while left < right:
                tmp_city = child[left]
                child[left] = child[right]
                child[right] = tmp_city
                left += 1
                right -= 1

            if two_opt_passes > 0:
                child = scalar_two_opt_descent_numba(weighted_matrices[direction_idx], child, two_opt_passes)
            if relocate_passes > 0:
                child = scalar_relocate_descent_numba(weighted_matrices[direction_idx], child, relocate_passes)

            obj0 = _cycle_cost2(matrices[0], child)
            obj1 = _cycle_cost2(matrices[1], child)
            child_tours[child_idx, :] = child
            child_objs[child_idx, 0] = obj0
            child_objs[child_idx, 1] = obj1

            if obj0 < ideal0:
                ideal0 = obj0
            if obj1 < ideal1:
                ideal1 = obj1
            if obj0 > nadir0:
                nadir0 = obj0
            if obj1 > nadir1:
                nadir1 = obj1
            inv0 = 1.0 / max(1e-9, nadir0 - ideal0)
            inv1 = 1.0 / max(1e-9, nadir1 - ideal1)

            accepted_this_child = 0
            for pos in range(neighbor_count):
                replace_idx = neighbors[direction_idx, pos]
                child_scalar = _base_scalar2_numba(
                    obj0,
                    obj1,
                    weight0[replace_idx],
                    weight1[replace_idx],
                    ideal0,
                    ideal1,
                    inv0,
                    inv1,
                )
                current_scalar = _base_scalar2_numba(
                    objs[replace_idx, 0],
                    objs[replace_idx, 1],
                    weight0[replace_idx],
                    weight1[replace_idx],
                    ideal0,
                    ideal1,
                    inv0,
                    inv1,
                )
                attempts += 1
                if child_scalar - current_scalar <= 0.0:
                    pop[replace_idx, :] = child
                    objs[replace_idx, 0] = obj0
                    objs[replace_idx, 1] = obj1
                    accepted += 1
                    accepted_this_child += 1
                else:
                    rejected += 1
            if accepted_this_child == 0:
                current_rejection_streak += 1
                if current_rejection_streak > max_rejection_streak:
                    max_rejection_streak = current_rejection_streak
            else:
                current_rejection_streak = 0
        return (
            pop,
            objs,
            child_tours,
            child_objs,
            attempts,
            accepted,
            rejected,
            current_rejection_streak,
            max_rejection_streak,
            ideal0,
            ideal1,
            nadir0,
            nadir1,
        )

    def warmup_numba_kernels() -> None:
        matrix = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 0.0, 1.5, 2.5, 3.5],
                [2.0, 1.5, 0.0, 1.0, 2.0],
                [3.0, 2.5, 1.0, 0.0, 1.0],
                [4.0, 3.5, 2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        tour = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        scalar_two_opt_descent_numba(matrix, tour, 1)
        scalar_relocate_descent_numba(matrix, tour, 1)
        population = np.vstack([tour, np.array([0, 2, 1, 3, 4], dtype=np.int64)])
        matrices = np.stack([matrix, matrix])
        weighted = np.stack([matrix, matrix])
        objectives = np.array(
            [[_cycle_cost2(matrix, population[0]), _cycle_cost2(matrix, population[0])],
             [_cycle_cost2(matrix, population[1]), _cycle_cost2(matrix, population[1])]],
            dtype=np.float64,
        )
        ips_scalar_polish_epoch_numba(
            matrices,
            weighted,
            population,
            objectives,
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([[0, 1], [1, 0]], dtype=np.int64),
            1,
            0,
            1,
            1,
            1,
            float(objectives[:, 0].min()),
            float(objectives[:, 1].min()),
            float(objectives[:, 0].max()),
            float(objectives[:, 1].max()),
            0,
            0,
        )
        scalar_greedy_population_numba(
            matrices,
            weighted,
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            1.0,
            1.0,
            2,
            2,
            2,
            2,
            1,
            1,
        )
        two_opt_objectives_batch_numba(
            matrices,
            tour,
            float(objectives[0, 0]),
            float(objectives[0, 1]),
            np.array([[1, 3]], dtype=np.int64),
        )

else:

    def scalar_two_opt_descent_numba(matrix, tour, max_passes):  # type: ignore[no-untyped-def]
        raise RuntimeError("Numba is not available.")

    def scalar_relocate_descent_numba(matrix, tour, max_passes):  # type: ignore[no-untyped-def]
        raise RuntimeError("Numba is not available.")

    def ips_scalar_polish_epoch_numba(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Numba is not available.")

    def scalar_greedy_population_numba(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Numba is not available.")

    def two_opt_objectives_batch_numba(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("Numba is not available.")

    def warmup_numba_kernels() -> None:
        return None
