1. 重构 game_state：

   - tensor 版本应该有 grid_game_channel, x_direction_channel, y_direction_channel, game_over_channel
   - 把 dict 版本换成 class (**Done**)

2. 实现 experience replay
3. 实现 loss 图像的绘制

（可选）

1. 制作 .exe 文件
2. 可以在软件中调整学习率等参数
