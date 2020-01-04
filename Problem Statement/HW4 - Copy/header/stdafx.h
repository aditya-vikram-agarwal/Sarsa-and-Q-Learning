#pragma once

// This file has our include statements, so in other files we can just include this one.

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#define _USE_MATH_DEFINES 
#include <math.h>
#include <climits>
#include<string>

// Tools
#include "MathUtils.hpp"
#include "FourierBasis.hpp"

// Environments
#include "MountainCar.hpp"
#include "CartPole.hpp"
#include "Acrobot.hpp"
#include "Gridworld.hpp"

// Agents
#include "QLearning.hpp"
#include "Sarsa.hpp"