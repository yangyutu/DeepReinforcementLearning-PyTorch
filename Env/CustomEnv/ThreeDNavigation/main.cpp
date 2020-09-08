#include "ActiveParticle3DSimulator.h"

void testSim_Const() {
    ActiveParticle3DSimulator simulator("config.json", 1);



	int step = 1000;
	simulator.createInitialState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
	std::default_random_engine rand_generator;
	std::normal_distribution<double> rand_normal{ 0.0, 1.0 };

	for (auto i = 0; i < step; ++i) {
		std::vector<double> actions = { 0.0, 0.0};
		simulator.run(20, actions);
	}
	simulator.close();

}
void testSim_Const_Director() {
    ActiveParticle3DSimulator simulator("config.json", 1);



	int step = 100;
	simulator.createInitialState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
	std::default_random_engine rand_generator;
	std::normal_distribution<double> rand_normal{ 0.0, 1.0 };

	for (auto i = 0; i < step; ++i) {
		std::vector<double> actions = { 0.0, 0.0, 1.0 };
		simulator.run_given_director(100, actions, true);
	}
	simulator.close();

}

void testSim_Random() {
	ActiveParticle3DSimulator simulator("config.json", 1);



	int step = 10000;
	simulator.createInitialState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
	std::default_random_engine rand_generator;
	std::normal_distribution<double> rand_normal{ 0.0, 1.0 };

	for (auto i = 0; i < step; ++i) {
		std::vector<double> actions = { rand_normal(rand_generator), rand_normal(rand_generator) };
		simulator.run(100, actions);
	}
	simulator.close();

}

void testSim_Policy() {
	ActiveParticle3DSimulator simulator("config.json", 1);

	std::vector<double> target = { 100.0, 100.0, 0.0 };

	int step = 1000;
	simulator.createInitialState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);


	for (auto i = 0; i < step; ++i) {
		std::vector<double> actions;

		actions = simulator.steeringAction(target);
		simulator.run(100, actions);

	}
	simulator.close();

}

int main() {

	testSim_Const_Director();
	return 0;
}