import numpy as np
import random
import time
import matplotlib.pyplot as plt
import traceback

class UCBEvoGA_Solver_4x4x4:
    def __init__(self, unsolved, n=50, T=100, c=2.0, m=0.2, generations=10000):
        self.N = 4  # 4x4x4 격자
        self.Ns = 2  # 2x2x2 서브큐브
        # 3D 배열로 변환 (4x4x4)
        self.unsolved = unsolved.reshape((self.N, self.N, self.N))
        self.fixed = (self.unsolved != 0).astype(int)
        self.population_set = []
        self.Q = np.zeros(n)
        self.n = n
        self.T = T
        self.c = c
        self.m = m
        self.generations = generations

    def generate_initial_board(self):
        """서브큐브 내에서 중복 없이 초기 해를 생성"""
        board = np.array(self.unsolved).copy()
        
        def get_possible_numbers(x, y, z):
            """특정 위치에 대해 가능한 숫자들 반환"""
            used = set()
            
            # 같은 행에서 사용된 숫자
            used.update(board[x, y, :])
            # 같은 열에서 사용된 숫자
            used.update(board[x, :, z])
            # 같은 깊이에서 사용된 숫자
            used.update(board[:, y, z])
            
            # 같은 서브큐브에서 사용된 숫자
            start_x, start_y, start_z = 2 * (x // 2), 2 * (y // 2), 2 * (z // 2)
            subcube = board[start_x:start_x+2, start_y:start_y+2, start_z:start_z+2]
            used.update(subcube.flatten())
            
            possible = list(set(range(1, 5)) - used)
            return possible if possible else list(range(1, 5))  # 가능한 숫자가 없으면 모든 숫자 반환

        # 빈 셀들의 위치를 수집
        empty_cells = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if board[x, y, z] == 0:
                        empty_cells.append((x, y, z))
        
        # 빈 셀들을 무작위로 섞음
        random.shuffle(empty_cells)
        
        # 각 빈 셀에 대해 시도
        for x, y, z in empty_cells:
            if board[x, y, z] == 0:
                possible_nums = get_possible_numbers(x, y, z)
                random.shuffle(possible_nums)
                
                for num in possible_nums:
                    if num not in board[x, y, :] and \
                       num not in board[x, :, z] and \
                       num not in board[:, y, z]:
                        board[x, y, z] = num
                        break
                
                if board[x, y, z] == 0:  # 적절한 숫자를 찾지 못했다면
                    board[x, y, z] = random.randint(1, 4)  # 임의의 숫자 할당
        
        return board

    def fittness(self, candidate):
        """행, 열, 깊이 및 모든 방향의 2x2 서브블록 평가"""
        if isinstance(candidate, list):
            candidate = np.array(candidate)
        
        candidate = candidate.reshape((self.N, self.N, self.N))
        total_score = 0
        
        # 행 평가
        for x in range(self.N):
            for y in range(self.N):
                row = list(candidate[x, y, :])
                total_score += len(set(row))/4
        
        # 열 평가
        for x in range(self.N):
            for z in range(self.N):
                col = list(candidate[x, :, z])
                total_score += len(set(col))/4
        
        # 깊이 평가
        for y in range(self.N):
            for z in range(self.N):
                depth = list(candidate[:, y, z])
                total_score += len(set(depth))/4
        
        # 수직 방향 2x2 서브블록 평가 (xz 평면, 총 16개)
        for x in range(0, 4, 2):
            for y in range(4):
                for z in range(0, 4, 2):
                    block = list(candidate[x:x+2, y, z:z+2].flatten())
                    total_score += len(set(block))/4
        
        # 측면 방향 2x2 서브블록 평가 (yz 평면, 총 16개)
        for x in range(4):
            for y in range(0, 4, 2):
                for z in range(0, 4, 2):
                    block = list(candidate[x, y:y+2, z:z+2].flatten())
                    total_score += len(set(block))/4
        
        # 총 제약조건:
        # - 16개 행
        # - 16개 열
        # - 16개 깊이
        # - 16개 수직 서브블록 (xz 평면)
        # - 16개 측면 서브블록 (yz 평면)
        # 총 80개의 제약조건
        return total_score / 80

    def reduce_constraint(self, unsolved):
        """수평 방향의 서브큐브 제약조건을 반드시 만족하는 초기해 생성"""
        board = np.array(self.unsolved).copy()
        
        # 각 층에 대해
        for z in range(4):
            # 해당 층의 2x2 서브큐브들을 처리
            for start_x in range(0, 4, 2):
                for start_y in range(0, 4, 2):
                    # 현재 서브큐브의 고정된 숫자들 찾기
                    used_numbers = set()
                    empty_positions = []
                    
                    # 2x2 서브큐브 내의 고정된 숫자와 빈 위치 찾기
                    for i in range(2):
                        for j in range(2):
                            x, y = start_x + i, start_y + j
                            val = board[x, y, z]
                            if val != 0:
                                used_numbers.add(val)
                            else:
                                empty_positions.append((x, y))
                    
                    # 남은 숫자들 (1-4 중에서 사용되지 않은 숫자들)
                    remaining_numbers = list(set(range(1, 5)) - used_numbers)
                    if len(remaining_numbers) < len(empty_positions):
                        # 남은 숫자가 부족하면 다시 시도
                        return self.reduce_constraint(unsolved)
                    
                    random.shuffle(remaining_numbers)
                    
                    # 빈 위치에 남은 숫자들 할당
                    for x, y in empty_positions:
                        board[x, y, z] = remaining_numbers.pop()
        
        return board

    def check_subcube_constraint(self, board, start_x, start_y, start_z):
        """특정 서브큐브가 제약조건을 만족하는지 확인"""
        subcube = board[start_x:start_x+2, start_y:start_y+2, start_z:start_z+2]
        values = set(subcube.flatten())
        # 0이 있거나 1-4가 아닌 값이 있으면 False
        if 0 in values or any(v > 4 or v < 1 for v in values):
            return False
        # 서브큐브 내의 모든 값이 서로 달라야 함
        return len(values) == 4

    def crossover(self, parent1, parent2):
        """서브큐브 단위로만 교차하고 서브큐브 제약을 엄격하게 유지"""
        parent1 = np.array(parent1).reshape((4, 4, 4))
        parent2 = np.array(parent2).reshape((4, 4, 4))
        child = np.zeros((4, 4, 4))
        
        subcube_starts = [(0,0,0), (0,0,2), (0,2,0), (0,2,2),
                          (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
        
        for start_x, start_y, start_z in subcube_starts:
            # 각 부모의 서브큐브가 제약을 만족하는지 확인
            p1_valid = self.check_subcube_constraint(parent1, start_x, start_y, start_z)
            p2_valid = self.check_subcube_constraint(parent2, start_x, start_y, start_z)
            
            if p1_valid and (not p2_valid or random.random() < 0.5):
                child[start_x:start_x+2, start_y:start_y+2, start_z:start_z+2] = \
                    parent1[start_x:start_x+2, start_y:start_y+2, start_z:start_z+2]
            elif p2_valid:
                child[start_x:start_x+2, start_y:start_y+2, start_z:start_z+2] = \
                    parent2[start_x:start_x+2, start_y:start_y+2, start_z:start_z+2]
            else:
                # 새로운 유효한 서브큐브 생성
                used_numbers = set()
                empty_positions = []
                
                # 고정된 숫자 찾기
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            x, y, z = start_x + i, start_y + j, start_z + k
                            if self.fixed[x, y, z]:
                                used_numbers.add(self.unsolved[x, y, z])
                            else:
                                empty_positions.append((x, y, z))
                
                remaining = list(set(range(1, 5)) - used_numbers)
                random.shuffle(remaining)
                
                # 빈 위치 채우기
                for pos in empty_positions:
                    if remaining:
                        child[pos] = remaining.pop()
                    else:
                        # 남은 숫자가 없으면 다시 시도
                        return self.crossover(parent1, parent2)
        
        return child.tolist()

    def mutation(self, child, mutation_rate, fixed):
        """서브큐브 내에서만 값을 교환하여 서브큐브 제약을 엄격하게 유지"""
        child = np.array(child).reshape((4, 4, 4))
        
        subcube_starts = [(0,0,0), (0,0,2), (0,2,0), (0,2,2),
                          (2,0,0), (2,0,2), (2,2,0), (2,2,2)]
        
        for start_x, start_y, start_z in subcube_starts:
            if random.random() < mutation_rate:
                # 현재 서브큐브의 값들 수집
                subcube_values = {}  # 위치: 값 매핑
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            x, y, z = start_x + i, start_y + j, start_z + k
                            if not fixed[x, y, z]:
                                subcube_values[(x, y, z)] = child[x, y, z]
                
                # 변경 가능한 위치가 2개 이상이면
                positions = list(subcube_values.keys())
                if len(positions) >= 2:
                    # 두 위치를 선택하여 값 교환
                    pos1, pos2 = random.sample(positions, 2)
                    val1, val2 = subcube_values[pos1], subcube_values[pos2]
                    
                    # 교환 후 서브큐브 제약이 유지되는지 확인
                    child[pos1], child[pos2] = val2, val1
                    if not self.check_subcube_constraint(child, start_x, start_y, start_z):
                        # 제약을 위반하면 원상복구
                        child[pos1], child[pos2] = val1, val2
        
        return child.tolist()

    def subblock(self, parent):
        """3D 서브큐브 추출"""
        parent = np.array(parent).reshape((self.N, self.N, self.N))
        subblock_list = []
        
        for i in range(2):  # x 축
            for j in range(2):  # y 축
                for k in range(2):  # z 축
                    # 2x2x2 서브큐브 추출
                    start_x = i * self.Ns
                    start_y = j * self.Ns
                    start_z = k * self.Ns
                    subblock = parent[start_x:start_x+self.Ns, 
                                    start_y:start_y+self.Ns,
                                    start_z:start_z+self.Ns]
                    subblock_list.append(subblock.flatten())
        
        return subblock_list

    def subblock_concate(self, subblock_list):
        """3D 서브큐브 재결합"""
        result = np.zeros((self.N, self.N, self.N))
        idx = 0
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    start_x = i * self.Ns
                    start_y = j * self.Ns
                    start_z = k * self.Ns
                    result[start_x:start_x+self.Ns,
                          start_y:start_y+self.Ns,
                          start_z:start_z+self.Ns] = np.array(subblock_list[idx]).reshape(self.Ns, self.Ns, self.Ns)
                    idx += 1
        
        return result

    def get_candidates(self, unsolved, x, y, z):
        """특정 위치에 대한 가능한 후보 숫자들 반환"""
        candidates = set(range(1, 5))
        
        # 같은 행에서 사용된 숫자 제거
        for i in range(self.N):
            if unsolved[x][y][i] in candidates:
                candidates.remove(unsolved[x][y][i])
        
        # 같은 열에서 사용된 숫자 제거
        for i in range(self.N):
            if unsolved[x][i][z] in candidates:
                candidates.remove(unsolved[x][i][z])
        
        # 같은 깊이에서 사용된 숫자 제거
        for i in range(self.N):
            if unsolved[i][y][z] in candidates:
                candidates.remove(unsolved[i][y][z])
        
        # 같은 서브큐브에서 사용된 숫자 제거
        start_x = (x // self.Ns) * self.Ns
        start_y = (y // self.Ns) * self.Ns
        start_z = (z // self.Ns) * self.Ns
        
        for i in range(start_x, start_x + self.Ns):
            for j in range(start_y, start_y + self.Ns):
                for k in range(start_z, start_z + self.Ns):
                    if unsolved[i][j][k] in candidates:
                        candidates.remove(unsolved[i][j][k])
        
        return list(candidates)

    def solve(self):
        """유전 알고리즘 실행"""
        print("초기 population 생성 중...")
        
        def initialize_population():
            population = []
            q_values = np.zeros(self.n)
            while len(population) < self.n:
                board = self.reduce_constraint(self.unsolved)
                board_list = board.tolist()
                if not any(np.array_equal(board, np.array(p)) for p in population):
                    population.append(board_list)
                    q_values[len(population)-1] = self.fittness(board)
            return population, q_values
        
        # 초기 population 생성
        self.population_set, self.Q = initialize_population()
        
        best_fitness = 0
        best_solution = None
        generation = 0
        stagnation_count = 0
        
        while generation < self.generations:
            # 자식 해 생성
            offspring = []
            while len(offspring) < self.n:
                p1, p2 = random.sample(self.population_set, 2)
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
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # 완벽한 해 확인
            if best_fitness >= 0.9999:
                print("완벽한 해를 찾았습니다!")
                return best_solution.reshape((4, 4, 4)).tolist()
            
            # 정체 상태가 오래 지속되면 population 완전 초기화
            if stagnation_count >= 100:
                print("정체 상태 감지. Population 완전 초기화...")
                # population을 완전히 새로 생성
                self.population_set, self.Q = initialize_population()
                stagnation_count = 0
                continue
            
            self.population_set = offspring
            self.Q = np.array(offspring_fitness)
            generation += 1
        
        return best_solution.reshape((4, 4, 4)).tolist() if best_solution is not None else None

if __name__ == "__main__":
    # 테스트용 4x4x4 스도쿠 퍼즐
    unsolved = np.zeros((4, 4, 4), dtype=int)
    
    # 1층 (z=0)
    unsolved[0, 0, 0] = 0
    unsolved[0, 1, 0] = 0
    unsolved[0, 2, 0] = 0
    unsolved[0, 3, 0] = 0
    unsolved[1, 0, 0] = 3
    unsolved[1, 1, 0] = 2
    unsolved[1, 2, 0] = 1
    unsolved[1, 3, 0] = 0
    unsolved[2, 0, 0] = 4
    unsolved[2, 1, 0] = 0
    unsolved[2, 2, 0] = 0
    unsolved[2, 3, 0] = 0
    unsolved[3, 0, 0] = 2
    unsolved[3, 1, 0] = 3
    unsolved[3, 2, 0] = 4
    unsolved[3, 3, 0] = 1

    # 2층 (z=1)
    unsolved[0, 0, 1] = 0
    unsolved[0, 1, 1] = 0
    unsolved[0, 2, 1] = 4
    unsolved[0, 3, 1] = 1
    unsolved[1, 0, 1] = 4
    unsolved[1, 1, 1] = 1
    unsolved[1, 2, 1] = 2
    unsolved[1, 3, 1] = 3
    unsolved[2, 0, 1] = 3
    unsolved[2, 1, 1] = 2
    unsolved[2, 2, 1] = 1
    unsolved[2, 3, 1] = 0
    unsolved[3, 0, 1] = 1
    unsolved[3, 1, 1] = 4
    unsolved[3, 2, 1] = 3
    unsolved[3, 3, 1] = 2

    # 3층 (z=2)
    unsolved[0, 0, 2] = 4
    unsolved[0, 1, 2] = 2
    unsolved[0, 2, 2] = 0
    unsolved[0, 3, 2] = 0
    unsolved[1, 0, 2] = 1
    unsolved[1, 1, 2] = 0
    unsolved[1, 2, 2] = 4
    unsolved[1, 3, 2] = 0
    unsolved[2, 0, 2] = 2
    unsolved[2, 1, 2] = 4
    unsolved[2, 2, 2] = 0
    unsolved[2, 3, 2] = 0
    unsolved[3, 0, 2] = 3
    unsolved[3, 1, 2] = 1
    unsolved[3, 2, 2] = 2
    unsolved[3, 3, 2] = 4

    # 4층 (z=3)
    unsolved[0, 0, 3] = 3
    unsolved[0, 1, 3] = 0
    unsolved[0, 2, 3] = 2
    unsolved[0, 3, 3] = 0
    unsolved[1, 0, 3] = 0
    unsolved[1, 1, 3] = 4
    unsolved[1, 2, 3] = 3
    unsolved[1, 3, 3] = 1
    unsolved[2, 0, 3] = 0
    unsolved[2, 1, 3] = 0
    unsolved[2, 2, 3] = 4
    unsolved[2, 3, 3] = 2
    unsolved[3, 0, 3] = 0
    unsolved[3, 1, 3] = 2
    unsolved[3, 2, 3] = 0
    unsolved[3, 3, 3] = 3
    
    print("초기 퍼즐:")
    for z in range(4):
        print(f"\n층 {z+1}:")
        print(unsolved[:, :, z])

    start_time = time.time()
    solver = UCBEvoGA_Solver_4x4x4(unsolved)
    solution = solver.solve()
    end_time = time.time()

    if solution is not None:
        print("\n해결된 퍼즐:")
        solution_array = np.array(solution)
        for z in range(4):
            print(f"\n층 {z+1}:")
            print(solution_array[:, :, z])
        print(f"\n소요 시간: {end_time - start_time:.2f}초")
        print(f"최종 적합도: {solver.fittness(solution):.4f}")
    else:
        print("\n해결책을 찾지 못했습니다.")