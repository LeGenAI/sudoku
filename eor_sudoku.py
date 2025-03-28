import numpy as np
import numpy.random as rnd
import random
import math
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('TkAgg')

class UCBEvoGA_Solver:
    def __init__(self, unsolved, n, T, c, m, generations):
        self.N = int(np.sqrt(len(unsolved)))
        self.Ns = int(np.sqrt(self.N))
        self.fixed = (unsolved != 0).astype(int)
        self.Q = np.zeros(n)
        self.population_set = []
        self.best_solution = []
        self.n = n
        self.T = T
        self.c = c
        self.m = m
        self.generations = generations
        self.cum_regret = []
        self.cumulative_regret = 0

    def investigation(self, unsolved):
        fixed = (unsolved != 0).astype(int)
        return fixed

    def get_candidates(self, unsolved, row, col):
        candidates = set(range(1, 10))

        for i in range(9):
            if unsolved[row][i] in candidates:
                candidates.remove(unsolved[row][i])
        #print("after row remove=", candidates)

        for i in range(9):
            if unsolved[i][col] in candidates:
                candidates.remove(unsolved[i][col])
        #print("after col remove=", candidates)

        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if unsolved[i][j] in candidates:
                    candidates.remove(unsolved[i][j])
        #print("after sub remove=", candidates)

        return list(candidates)

    def fill_board(self, unsolved):
        unsolved_copy = unsolved.copy().reshape((self.N, self.N))

        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                subblock_candidates = set(range(1, 10))
                for i in range(row, row + 3):
                    for j in range(col, col + 3):
                        if unsolved_copy[i][j] in subblock_candidates:
                            subblock_candidates.remove(unsolved_copy[i][j])

                # 서브블록 내에 후보 숫자가 없는 경우
                if not subblock_candidates:
                    return False

                for i in range(row, row + 3):
                    for j in range(col, col + 3):
                        if unsolved_copy[i][j] == 0:
                            random_candidate = random.choice(list(subblock_candidates))
                            unsolved_copy[i][j] = random_candidate
                            subblock_candidates.remove(random_candidate)

        return unsolved_copy

    def reduce_constraint(self, unsolved):
        unsolved = unsolved.reshape((self.N, self.N))
        c = self.fill_board(unsolved)
        c = np.reshape(c, (self.N, self.N))

        return c

    def fittness(self, candidate):
        candidate = np.array(candidate)

        evaluation_row = 0
        for irow in range(self.N):
            row = candidate[irow, :]
            row_diversity_set = set(row)
            evaluation_row += len(row_diversity_set)

        evaluation_col = 0
        for icol in range(self.N):
            col = candidate[:, icol]
            col_diversity_set = set(col)
            evaluation_col += len(col_diversity_set)

        fittness = (evaluation_row + evaluation_col) / (2 * self.N * self.N)

        return fittness

    def subblock(self, parent):
        parent_copy = parent.reshape((self.N, self.N))
        subblock_list = []

        for i in range(self.Ns):
            for j in range(self.Ns):
                # Extract subblocks
                subblock = parent_copy[i * self.Ns:(i + 1) * self.Ns, j * self.Ns:(j + 1) * self.Ns]
                subblock = subblock.flatten()
                subblock_list.append(subblock)

        return subblock_list

    def subblock_concate(self, subblock_list):
        almost = []
        for i in range(self.N):
            sub = np.array(subblock_list[i]).reshape(self.Ns, self.Ns)
            almost.append(sub)
        subrow_ls = []
        for i in range(self.Ns):
            sub_row = np.concatenate(almost[i * self.Ns: (i + 1) * self.Ns], axis=1)
            subrow_ls.append(sub_row)
        candidate = np.concatenate(subrow_ls[:], axis=0)
        candidate = np.reshape(candidate, (self.N, self.N))

        return candidate

    def crossover(self, parent1, parent2):
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)
        p1 = self.subblock(parent1)
        p2 = self.subblock(parent2)
        cross_point = list(range(0, 9))
        rnd.shuffle(cross_point)
        cross_result = [0] * self.N

        for i in range(self.N // 2):
            cross_result[cross_point[i]] = p1[cross_point[i]]
        for i in range(self.N // 2, self.N):
            cross_result[cross_point[i]] = p2[cross_point[i]]

        child = self.subblock_concate(cross_result)

        return child

    def ucbbandit(self, n, T, c):
        self.cum_regret = []
        self.cumulative_regret = 0
        ucb_values = np.zeros(self.n)
        arm_pulls = np.zeros(self.n)
        candidate_idx_pairs = []
        population_index = list(range(self.n))

        while len(population_index) >= 2:
            candidate1_idx, candidate2_idx = random.sample(population_index, 2)
            population_index.remove(candidate1_idx)
            population_index.remove(candidate2_idx)
            candidate_idx_pairs.append([candidate1_idx, candidate2_idx])

        for t in range(self.T):

            idx_1 = candidate_idx_pairs[t][0]
            idx_2 = candidate_idx_pairs[t][1]

            parent1 = self.population_set[idx_1]
            parent2 = self.population_set[idx_2]

            child1 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1, self.m, self.fixed)
            parent1 = np.reshape(parent1, (self.N, self.N))
            parent2 = np.reshape(parent2, (self.N, self.N))

            fp1 = self.fittness(parent1)
            fp2 = self.fittness(parent2)
            new_fittness = self.fittness(child1)
            optimal_reward = 1  # 최적 액션 선택
            self.cumulative_regret += 1 - new_fittness
            self.cum_regret.append(self.cumulative_regret)

            new_fittness_2 = new_fittness - fp1
            new_fittness_1 = new_fittness - fp2

            arm_pulls[idx_1] += 1
            arm_pulls[idx_2] += 1

            self.Q[idx_1] = (self.Q[idx_1] + new_fittness_1) / arm_pulls[idx_1]
            self.Q[idx_2] = (self.Q[idx_2] + new_fittness_2) / arm_pulls[idx_2]

            ucb_values[idx_1] = self.Q[idx_1] + c * (math.log(t + 1) / arm_pulls[idx_1]) ** 0.5
            ucb_values[idx_2] = self.Q[idx_2] + c * (math.log(t + 1) / arm_pulls[idx_2]) ** 0.5

            return ucb_values

    def mutation(self, child, m, fixed):
        while True:
            s = random.randint(0, 8)
            subblock_size = self.N // 3
            subblock_row = s // subblock_size
            subblock_col = s % subblock_size

            row1_in_subblock = random.randint(0, subblock_size - 1)
            col1_in_subblock = random.randint(0, subblock_size - 1)

            row1 = subblock_row * subblock_size + row1_in_subblock
            col1 = subblock_col * subblock_size + col1_in_subblock

            row2_in_subblock = random.randint(0, subblock_size - 1)
            col2_in_subblock = random.randint(0, subblock_size - 1)
            while row1_in_subblock == row2_in_subblock and col1_in_subblock == col2_in_subblock:
                row2_in_subblock = random.randint(0, subblock_size - 1)
                col2_in_subblock = random.randint(0, subblock_size - 1)

            row2 = subblock_row * subblock_size + row2_in_subblock
            col2 = subblock_col * subblock_size + col2_in_subblock

            fixed = np.reshape(fixed, (self.N, self.N))

            if fixed[row1][col1] == 0 and fixed[row2][col2] == 0:
                # 선택된 셀이 모두 고정된 숫자가 아닌 경우에만 교환
                child[row1][col1], child[row2][col2] = child[row2][col2], child[row1][col1]
                break

        return child

    def solve(self):
        p = []
        while len(self.population_set) < self.n:
            b = self.reduce_constraint(unsolved)
            candi = b.tolist()

            if candi not in self.population_set:
                self.population_set.append(candi)
                p.append(b)


        for i in range(self.n):
            a = self.fittness(self.population_set[i])
            self.Q[i] = a
            if int(a) == 1:
                self.population_set[i] = np.array(self.population_set[i])
                self.population_set[i] = np.reshape(self.population_set[i], (self.N, self.N))
                self.best_solution.append(self.population_set[i])

                print("Solution is\n", self.best_solution)
                print("The best fittness is ", k)

                return self.best_solution

        again = 0
        repetition = 0
        num = 0

        for iteration in range(1, self.generations + 1):
            child_set = []
            child_fittness = []
            temp_set = []
            temp_fittness = []

            ucb = self.ucbbandit(self.n, self.T, self.c)
            k = self.n
            top_k_indices = np.argsort(ucb)[-k:][::-1]
            temp_set = [self.population_set[i] for i in top_k_indices]

            # temp_set의 각 원소에 대한 적합도를 계산
            fitness_scores = [self.fittness(individual) for individual in temp_set]
            # 적합도가 큰 순서대로 정렬된 인덱스를 가져옴
            sorted_indices = np.argsort(fitness_scores)[::-1]

            # 상위 k//3개의 개체를 선택
            elite = [temp_set[i] for i in sorted_indices[:k // 2]]

            while len(child_set) < self.n:
                p1 = random.randint(0, k // 2 - 1)
                p2 = random.randint(0, k // 2 - 1)
                parent1 = elite[p1]
                parent2 = elite[p2]

                child = self.crossover(parent1, parent2)
                new = self.fittness(child)
                if int(new) != 1:
                    prob = rnd.uniform()
                    if prob < self.m:
                        child = self.mutation(child, self.m, self.fixed)
                        child = self.mutation(child, self.m, self.fixed)
                        child = self.mutation(child, self.m, self.fixed)
                        child = self.mutation(child, self.m, self.fixed)
                        child = self.mutation(child, self.m, self.fixed)

                    new = self.fittness(child)

                if not any(np.array_equal(child, item) for item in child_set):
                    child_set.append(child)
                    child_fittness.append(new)


            best = child_fittness.index(max(child_fittness))

            child_set[best] = np.array(child_set[best])
            child_set[best] = np.reshape(child_set[best], (self.N, self.N))

            self.best_solution.append(child_set[best])
            self.population_set = child_set
            self.Q = child_fittness

            if again == 0:  # 적합도 반복횟수
                again = again + child_fittness[best]
            elif again == child_fittness[best]:
                repetition += 1
            else:
                repetition = 0
                again = 0

            if int(child_fittness[best]) == 1:
                print("아싸 가오리")
                print("solution is\n", child_set[best])
                child_set[best] = np.array(child_set[best])
                child_set[best] = np.reshape(child_set[best], (self.N, self.N))
                self.best_solution.append(child_set[best])
                return self.best_solution

            if repetition == 10:
                if num < 10:
                    p = 0
                    for i in range(len(child_set)):
                        while p > 30:
                            child_set[i] = self.mutation(child_set[i], self.m, self.fixed)
                            p += 1

                        new = self.fittness(child_set[i])
                        child_fittness[i] = new

                        child_set[i] = np.array(child_set[i])
                        child_set[i] = np.reshape(child_set[i], (self.N, self.N))

                    self.population_set = child_set
                    self.Q = child_fittness
                    num += 1
                    repetition = 0
                    again = 0

                elif num == 10:
                    num = 0
                    print("Re-seeding is needed")
                    repetition = 0
                    again = 0
                    self.population_set = []
                    self.Q = np.zeros(self.n)

                    while len(self.population_set) < self.n:
                        b = self.reduce_constraint(unsolved)
                        candi = b.tolist()
                        self.population_set.append(candi)

                    for i in range(self.n - 1):
                        a = self.fittness(self.population_set[i])
                        self.Q[i] = a

                        if int(a) == 1:
                            self.population_set[i] = np.array(self.population_set[i])
                            self.population_set[i] = np.reshape(self.population_set[i], (self.N, self.N))
                            self.best_solution.append(self.population_set[i])

                            print("Solution is\n", self.best_solution)
                            print("The best fittness is ", k)
                            return self.best_solution

                    self.population_set.append(child_set[best])

        return self.best_solution


# Example Usage: easy1
unsolved = np.array(
    [0, 2, 3, 0, 8, 0, 0, 0, 7, 5, 0, 7, 0, 3, 9, 0, 0, 0, 0, 9, 4, 0, 1, 2, 5, 0, 0, 0, 0, 1, 6, 0, 0, 0, 3, 0, 0, 5, 0, 0, 9, 0, 0, 2, 0, 0, 6, 0, 0, 0, 1, 7, 0, 0, 0, 0, 6, 2, 5, 0, 3, 7, 0, 0, 0, 0, 1, 7, 0, 2, 0, 6, 7, 0, 0, 0, 6, 0, 4, 1, 0]
)

import time
start=time.time()
solver = UCBEvoGA_Solver(unsolved, 150,500, 2.0, 0.5, 10000)
# class UCBEvoGA_Solver(self, unsolved, n, T, c, m, generations):
best_solution = solver.solve()

best_fittness_values = []  # 각 세대의 최적 적합도를 저장할 리스트
generation_numbers = []  # 각 세대의 번호를 저장할 리스트

# best_solution = np.reshape(best_solution, (9, 9))

for generation, solution in enumerate(best_solution, start=1):
    best_fittness = solver.fittness(solution)
    best_fittness_values.append(best_fittness)
    generation_numbers.append(generation)

print("best_fittness_values =", best_fittness_values)

# 꺾은선 그래프
plt.plot(generation_numbers, best_fittness_values, color='green', linestyle='solid')
plt.title('Best Fittness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fittness')
plt.grid(True)
plt.show()

end=time.time()
print("time=",end-start)