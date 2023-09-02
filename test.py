import copy

a = [10, 20, 20, 30, 50, 50]

def find_solution(current_a, target_value, notes):
    if notes == 1:
        #print("---- base case", current_a, target_value)
        ways = 0
        for c in current_a:
            if c == target_value:
                ways += 1
        return ways
    
    ways = 0
    for idx, c in enumerate(current_a[:len(current_a)-1]):
        #print("main case", c, current_a, target_value)
        new_target = target_value - c

        if target_value > 0 and notes > 1:
            w = find_solution(current_a[idx+1:], new_target, notes-1)
            #print("-- main case", c, current_a[idx+1:], new_target, w)
            ways += w 
    return ways       


for i in range(1,len(a)+1):
    print(i,"notes",find_solution(a, 120, i))