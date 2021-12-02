
#  Control of mobile robots using Double QPID algorithm.

This project is the implementation of the paper by Ignacio Carlucho on ["Double Q-learning algorithm for mobile robot control"](https://www.sciencedirect.com/science/article/pii/S0957417419304749).

We have presented a technical report on this paper which you can find it [here](https://github.com/karanamrahul/DoubleQPID-Algorithm/blob/main/DoubleQPID_report.pdf).

# Scope

An expert agent- based system, based on a reinforcement learning agent, for self-adapting multiple low-level PID controllers in mobile robots.


We are demonstrating our implementation by using husky and hector quadrotor.


## Requirements

ROS Melodic or Noetic

Python 2.7 or higher

numpy

OpenAI gym

## Robots used for simulation

### Husky

Download the required dependancies for the Husky robot from [here](http://wiki.ros.org/Robots/Husky).

To simulate husky, first we need to download all the required dependancies for husky and then launch your husky in gazebo using the below launch file. After launching the husky in gazebo, set the robot platform to husky and run the main.py program to simulate the Double QPID algorithm.

```bash
roslaunch husky_gazebo husky_empty_world.launch
```

``` set
platform = 'husky_pi_random'
```
```
python main.py
```

### Hector Quadrotor
Download the required dependancies for the Husky robot from [here](https://github.com/RAFALAMAO/hector_quadrotor_noetic).


Similarly, to simulate the hector quadrotor, we follow the same instructions as followed by the husky.

```bash
roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch 
```

``` set
platform = 'drone_pi'
```
```
python main.py
```

## Personell 
```bash
Sumedh Reddy Koppula
Master's student, University of Maryland, College Park

Rahul Karanam
Master's student, University of Maryland, College Park
```

Results

### Husky

![](https://github.com/karanamrahul/DoubleQPID-Algorithm/blob/main/Husky/Results/huskysim.png)


### Hector Quadrotor
![](https://github.com/karanamrahul/DoubleQPID-Algorithm/blob/main/HectorQ/Results/Drone.png)

## References

1.[Reinforcement Q-learning PID Controller for a Restaurant Mobile Robot](https://dl.acm.org/doi/10.1145/3380688.3380718)

2.[Incremental Q-learning strategy for adaptive PID control of mobile robots](https://www.sciencedirect.com/science/article/pii/S0957417417301513)


