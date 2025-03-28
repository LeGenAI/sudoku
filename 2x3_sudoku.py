import numpy as np
import random
import time
import matplotlib.pyplot as plt

class UCBEvoGA_Solver_6x6:
    def __init__(self, unsolved, n=50, T=100, c=2.0, m=0.1, generations=1000):
        self.N = 6  # 6x6 격자
        self.Ns_row = 2  # 서브블록 행 크기
        self.Ns_col = 3  # 서브블록 열 크기
        self.unsolved = unsolved
        self.fixed = (unsolved != 0).astype(int)
        self.population_set = []
        self.Q = np.zeros(n)
        self.n = n
        self.T = T
        self.c = c
        self.m = m
        self.generations = generations

    def generate_initial_board(self):
        """서브블록 내에서 중복 없이 초기 해를 생성"""
        board = np.array(self.unsolved).reshape((self.N, self.N)).copy()
        
        # 각 2x3 서브블록을 순회
        for i in range(3):  # 서브블록 행 인덱스 (0, 1, 2)
            for j in range(2):  # 서브블록 열 인덱스 (0, 1)
                start_row = i * self.Ns_row
                start_col = j * self.Ns_col
                
                # 현재 서브블록의 고정된 숫자들 수집
                fixed_vals = set()
                empty_positions = []
                
                for r in range(start_row, start_row + self.Ns_row):
                    for c in range(start_col, start_col + self.Ns_col):
                        if board[r][c] != 0:
                            fixed_vals.add(board[r][c])
                        else:
                            empty_positions.append((r, c))
                
                # 채워야 할 숫자들 (1-6 중 고정된 숫자 제외)
                available_nums = list(set(range(1, 7)) - fixed_vals)
                random.shuffle(available_nums)
                
                # 빈 위치에 숫자 채우기
                for pos, num in zip(empty_positions, available_nums):
                    board[pos[0]][pos[1]] = num
                    
        return board

    def fittness(self, candidate):
        if isinstance(candidate, list):
            candidate = np.array(candidate)
        
        if len(candidate.shape) == 1:
            candidate = candidate.reshape((self.N, self.N))
        
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

    def crossover(self, parent1, parent2):
        """서브블록 단위로만 교차를 수행"""
        parent1 = np.array(parent1).reshape((self.N, self.N))
        parent2 = np.array(parent2).reshape((self.N, self.N))
        child = np.copy(parent1)
        
        for i in range(3):  # 3개의 서브블록 행
            for j in range(2):  # 2개의 서브블록 열
                if random.random() < 0.5:
                    start_row = i * self.Ns_row
                    start_col = j * self.Ns_col
                    child[start_row:start_row+self.Ns_row, 
                         start_col:start_col+self.Ns_col] = \
                        parent2[start_row:start_row+self.Ns_row, 
                               start_col:start_col+self.Ns_col]
        
        child = np.where(self.fixed.reshape((self.N, self.N)), 
                        self.unsolved.reshape((self.N, self.N)), 
                        child)
        return child.tolist()

    def mutation(self, child, mutation_rate, fixed):
        """서브블록 내에서만 돌연변이를 수행"""
        child = np.array(child).reshape((self.N, self.N))
        mutated = np.copy(child)
        fixed = np.array(fixed).reshape((self.N, self.N))
        
        for i in range(3):      # 서브블록 행 인덱스
            for j in range(2):  # 서브블록 열 인덱스
                if random.random() < mutation_rate:
                    start_row = i * self.Ns_row
                    start_col = j * self.Ns_col
                    mutable_positions = []
                    
                    for r in range(start_row, start_row+self.Ns_row):
                        for c in range(start_col, start_col+self.Ns_col):
                            if not fixed[r][c]:
                                mutable_positions.append((r, c))
                    
                    if len(mutable_positions) >= 2:
                        pos1, pos2 = random.sample(mutable_positions, 2)
                        mutated[pos1[0]][pos1[1]], mutated[pos2[0]][pos2[1]] = \
                            mutated[pos2[0]][pos2[1]], mutated[pos1[0]][pos1[1]]
        
        return mutated.tolist()

    def solve(self):
        """유전 알고리즘 실행"""
        print("초기 population 생성 중...")
        
        # 초기 population 생성
        while len(self.population_set) < self.n:
            board = self.generate_initial_board()
            if board is not None:
                board_list = board.tolist()
                board_str = str(board_list)
                if not any(str(p) == board_str for p in self.population_set):
                    self.population_set.append(board_list)
                    self.Q[len(self.population_set)-1] = self.fittness(board)
        
        best_fitness = 0
        best_solution = None
        generation = 0
        stagnation_count = 0
        
        while True:  # 무한 루프로 변경
            # 상위 50% 선택
            sorted_indices = np.argsort(self.Q)[-self.n:][::-1]
            elite = [self.population_set[i] for i in sorted_indices[:self.n//2]]
            
            # 자식 해 생성
            offspring = []
            while len(offspring) < self.n:
                p1, p2 = random.sample(elite, 2)
                child = self.crossover(p1, p2)
                
                if random.random() < self.m:
                    child = self.mutation(child, self.m, self.fixed)
                offspring.append(child)
            
            # 평가 및 선택
            offspring_fitness = [self.fittness(child) for child in offspring]
            
            # 최고 적합도 갱신
            current_best = max(offspring_fitness)
            if current_best > best_fitness:
                best_fitness = current_best
                best_idx = offspring_fitness.index(current_best)
                best_solution = np.array(offspring[best_idx])
                print(f"세대 {generation}: 최적 적합도 = {best_fitness:.4f}")
                stagnation_count = 0  # 개선이 있으면 정체 카운트 리셋
            else:
                stagnation_count += 1
            
            # 완벽한 해 확인
            if best_fitness >= 0.9999:
                print("완벽한 해를 찾았습니다!")
                return best_solution.tolist()
            
            # 정체 상태가 오래 지속되면 population 재생성
            if stagnation_count >= 100:  # 100세대 동안 개선이 없으면
                print("정체 상태 감지. Population 재생성...")
                self.population_set = []
                self.Q = np.zeros(self.n)
                while len(self.population_set) < self.n:
                    board = self.generate_initial_board()
                    if board is not None:
                        board_list = board.tolist()
                        board_str = str(board_list)
                        if not any(str(p) == board_str for p in self.population_set):
                            self.population_set.append(board_list)
                            self.Q[len(self.population_set)-1] = self.fittness(board)
                stagnation_count = 0
                continue

            self.population_set = offspring
            self.Q = np.array(offspring_fitness)
            generation += 1

if __name__ == "__main__":
    unsolved0 = np.array([
        [4, 0, 0, 0, 0, 3],
        [0, 3, 0, 0, 0, 0],
        [2, 0, 3, 0, 0, 1],
        [0, 0, 0, 3, 2, 6],
        [0, 2, 1, 0, 6, 0],
        [5, 0, 6, 0, 3, 2]
    ])

    unsolved1 = np.array([
        [2, 0, 0, 1, 0, 6],
        [1, 6, 5, 0, 4, 2],
        [5, 2, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 1, 2, 3, 0]
    ])

    unsolved2 = np.array([
        [0, 4, 0, 0, 0, 0],
        [0, 0, 6, 2, 0, 0],
        [0, 5, 3, 0, 0, 6],
        [6, 0, 0, 5, 4, 0],
        [0, 0, 4, 3, 0, 0],
        [0, 0, 0, 0, 6, 0]
    ])

    unsolved = np.array([
        [3, 0, 0, 5, 0, 0],
        [0, 0, 0, 2, 0, 3],
        [4, 0, 0, 0, 0, 5],
        [2, 0, 0, 0, 0, 6],
        [6, 0, 4, 0, 0, 0],
        [0, 0, 5, 0, 0, 4]
    ])

    print("초기 퍼즐:")
    print(unsolved2)

    start_time = time.time()
    solver = UCBEvoGA_Solver_6x6(unsolved2)
    solution = solver.solve()
    end_time = time.time()

    print("\n해결된 퍼즐:")
    print(np.array(solution))
    print(f"\n소요 시간: {end_time - start_time:.2f}초")
    print(f"최종 적합도: {solver.fittness(solution):.4f}")