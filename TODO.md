1. 重构 game_state：

   - tensor 版本应该有 grid_game_channel, direction_channel, game_over_channel (**Done**)
   - 把 dict 版本换成 class (**Done**)
   - 去掉 direction_channel，在周围放边框，给一个 direction 的预览。需要修改 dqn_model 的输入和 game_elements 的 tensor 形态

2. 实现 experience replay
3. 实现 loss 图像的绘制

（可选）

1. 制作 .exe 文件
2. 可以在软件中调整学习率等参数
