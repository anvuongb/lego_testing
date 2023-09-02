
# v3 dev notes
- let's add more brick types, need to do:
    - prepare all possible actions, including rotation
    - that's it?
- allows for moving down too?
# v2 dev notes
- v2 allows training for 3-10 levels pyramid
- only allows 2x2 bricks
- still allows for moving up only
- [new] update big reward when fully complete a layer or entire model

# old notes
- test.ldr -> base mlp policy, no rewards for moveup
- test_2.ldr -> base mlp policy, rewards for moveup
- test_3.ldr -> mlp 128,128,128 with relu, rewards for moveup
- test_4.ldr -> mlp 128,128,128 with relu, rewards for moveup = total filled holes of current layer, no punishment for moving to early => failed
- test_5.ldr -> mlp 128,128,128 with relu, rewards for moveup, change perce to 0.6
- ^^^ all of these contains massive bug in next layer stud mat
- tried PPO and DQN with "full" placement mode, still not working, tends to stuck in some local minima
- need to recheck the occupancy mask
- improve placement list generation, check rand_tower_fix_base, bricks can now be place on imperfect surface (not all 0's)