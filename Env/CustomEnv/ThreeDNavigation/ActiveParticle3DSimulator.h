#pragma once
#include<vector>
#include<memory>
#include<random>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using json = nlohmann::json;
namespace py = pybind11;

enum class ParticleType {
    FULLCONTROL,
	VANILLASP,
	CIRCLER,
	SLIDER,
	TWODIM
};


struct ParticleState {
	double r[3], F[3];
	double orientVec[3];
	double oriMoveDirection[2][3];
	double theta, phi;
	double u, v, w;
	ParticleType type;
	ParticleState(double x = 0, double y = 0, double z = 0) {
		r[0] = x; r[1] = y; r[2] = z;
		orientVec[0] = 1.0;
		orientVec[1] = 0.0;
		orientVec[2] = 0.0;
		F[0] = 0.0;
		F[1] = 0.0;
		F[2] = 0.0;

		u = 1.0;
		v = 0;
		w = 0;
		updateAngles();
	}

	void cart2Angle() {

		double XsqPlusYsq = orientVec[0] * orientVec[0] + orientVec[1] * orientVec[1];
		theta = atan2(sqrt(XsqPlusYsq), orientVec[2]);
		phi = atan2(orientVec[1], orientVec[0]);
	}

	void normalizeOri() {
		double length = sqrt(orientVec[0] * orientVec[0] + orientVec[1] * orientVec[1] + orientVec[2] * orientVec[2]);
		for (int i = 0; i < 3; i++)
			orientVec[i] /= length;
	}

	void updateAngles() {
		normalizeOri();
		cart2Angle();

		oriMoveDirection[0][0] = -sin(phi);
		oriMoveDirection[0][1] = cos(phi);
		oriMoveDirection[0][2] = 0.0;

		oriMoveDirection[1][0] = (orientVec[1] * oriMoveDirection[0][2] - orientVec[2] * oriMoveDirection[0][1]);
		oriMoveDirection[1][1] = (orientVec[2] * oriMoveDirection[0][0] - orientVec[0] * oriMoveDirection[0][2]);
		oriMoveDirection[1][2] = (orientVec[0] * oriMoveDirection[0][1] - orientVec[1] * oriMoveDirection[0][0]);

	}


};

class ActiveParticle3DSimulator {
public:


	ActiveParticle3DSimulator(std::string configName, int randomSeed = 0);
	~ActiveParticle3DSimulator()
	{
		trajOs.close();
	}
	void runHelper();
	void run(int steps, const std::vector<double>& actions);
	void createInitialState(double x, double y, double z, double ori0, double ori1, double ori2);
	void calForces();
	void readConfigFile();
	void close();
	void step(int nstep, py::array_t<double>& actions);
	void setInitialState(double x, double y, double z, double ori0, double ori1, double ori2);
	py::array_t<double> get_positions();
	std::vector<double> get_positions_cpp();
	json config;
	std::vector<double> steeringAction(std::vector<double> target);
private:
	static const int dimP = 3;
	static const double kb, T, vis;
	int randomSeed;
	double maxSpeed, maxTurnSpeed, maxRotationSpeed;
	std::string configName;
	std::shared_ptr<ParticleState> particle;
	double radius, radius_nm;
	double Bpp; //2.29 is Bpp/a/kT
	double gravity;
	double Kappa; // here is kappa*radius
	double dt_, cutoff, mobility, diffusivity_r, diffusivity_t, Tc;
	std::default_random_engine rand_generator;
	std::normal_distribution<double> rand_normal{ 0.0, 1.0 };
	std::uniform_real_distribution<double> rand_uniform{ 0.0, 1.0 };

	int trajOutputInterval;
	long long timeCounter, fileCounter;
	std::ofstream trajOs;
	std::string filetag;
	bool trajOutputFlag, randomMoveFlag;
	void outputTrajectory(std::ostream& os);

};