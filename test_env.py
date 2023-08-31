from simple_env_v2 import SimpleLegoEnv, mask_fn

env = SimpleLegoEnv()

env.reset()
print(env.actions_map,"\n")
print("control stud")
print(env.action_control_stud_mat,"\n")
allowed_actions_idx = mask_fn(env)
allowed_actions = []
for idx, b in enumerate(allowed_actions_idx):
    if b:
        allowed_actions.append(env.all_actions[idx])
print(allowed_actions)
for idx, mat in enumerate(env.target_stud_mat_list):
    print("pyramid height", env.pyramid_levels)
    print("level", idx)
    print(mat,"\n")