#include "ActiveParticle3DSimulator.h"


double const ActiveParticle3DSimulator::T = 293.0;
double const ActiveParticle3DSimulator::kb = 1.38e-23;
double const ActiveParticle3DSimulator::vis = 1e-3;

ActiveParticle3DSimulator::ActiveParticle3DSimulator(std::string configName0, int randomSeed0) {
    randomSeed = randomSeed0;
	configName = configName0;
	std::ifstream ifile(this->configName);
	ifile >> config;
	ifile.close();
	readConfigFile();
}

void ActiveParticle3DSimulator::readConfigFile() {
	//auto iniConfig = config["iniConfig"];
	//double x = iniConfig[0];

	particle = std::make_shared<ParticleState>(0, 0, 0);

	randomMoveFlag = config["randomMoveFlag"];

	filetag = config["filetag"].get<std::string>();
	//std::vector<double> iniConfig;
	std::string typeString = config["particleType"];

	if (typeString.compare("VANILLASP") == 0) {
		particle->type = ParticleType::VANILLASP;
	}
	else if (typeString.compare("SLIDER") == 0) {
		particle->type = ParticleType::SLIDER;
	}
	else {
		std::cout << "particle type out of range" << std::endl;
		exit(2);
	}


	diffusivity_r = 0.161; // characteristic time scale is about 6s
        if (config.contains("Dr")) {
                diffusivity_r= config["Dr"];
        }        
        
	Tc = 1.0 / diffusivity_r;

	maxSpeed = config["maxSpeed"]; //units of radius per chacteristic time
	maxRotationSpeed = config["maxRotationSpeed"];
        
	radius = config["radius"];
	maxSpeed = maxSpeed * radius / Tc;
    if (config.contains("circularRadius")){
        circularRadius = config["circularRadius"];
        maxRotationSpeed = maxSpeed / circularRadius / radius;
        relaxationTimeInverse = maxSpeed / circularRadius / radius;
    }

	if (config.contains("ambientFieldVelocity")) {
		ambientFieldVelocity = config["ambientFieldVelocity"].get<std::vector<double>>();
		for (auto &e: ambientFieldVelocity)
			e = e * radius / Tc;
	}
        
    gravity = 0.0;
    if (config.contains("gravity")) {
	    gravity = config["gravity"];
	    gravity *= kb * T / radius;
    }

	dt_ = config["dt"]; // units of characteristc time
	trajOutputInterval = 1.0 / dt_;
	if (config.contains("trajOutputInterval")) {
		trajOutputInterval = config["trajOutputInterval"];
	}

	dt_ = dt_*Tc;
	diffusivity_t = 2.145e-13; // this corresponds the diffusivity of 1um particle
	diffusivity_t = 2.145e-14; // here I want to manually decrease the random noise
							   //diffusivity_r = parameter.diffu_r; // this correponds to rotation diffusity of 1um particle
	if (config.contains("Dt")) {
            diffusivity_t = config["Dt"];
	}
        
	Bpp = config["Bpp"];
	Bpp = Bpp * kb * T * 1e9; //2.29 is Bpp/a/kT
	Kappa = config["kappa"]; // here is kappa*radius
	radius_nm = radius * 1e9;
	mobility = diffusivity_t / kb / T;

	// attraction paramters
	//Os_pressure = config["Os_pressure"];
	//Os_pressure = Os_pressure * kb * T * 1e9;
	//L_dep = config["L_dep"]; // 0.2 of radius size, i.e. 200 nm
	//combinedSize = (1 + L_dep) * radius_nm;

	fileCounter = 0;
	cutoff = config["cutoff"];
	trajOutputFlag = config["trajOutputFlag"];
	this->rand_generator.seed(randomSeed);
}


py::array_t<double> ActiveParticle3DSimulator::get_positions() {

	std::vector<double> positions(6);

	positions = this->get_positions_cpp();

	py::array_t<double> result(6, positions.data());

	return result;

}

std::vector<double> ActiveParticle3DSimulator::get_positions_cpp() {
	std::vector<double> positions(6);
	positions[0] = particle->r[0] / radius;
	positions[1] = particle->r[1] / radius;
	positions[2] = particle->r[2] / radius;
	positions[3] = particle->orientVec[0];
	positions[4] = particle->orientVec[1];
	positions[5] = particle->orientVec[2];
	return positions;
}

void ActiveParticle3DSimulator::run_given_director(int steps, const std::vector<double>& director,bool local) {

	if (particle->type == ParticleType::VANILLASP) {
		exit(2);
	}

	if (particle->type == ParticleType::SLIDER) {
		particle->u = 1.0;
		particle->v = 0;
		particle->w = 0;
	}

        std::vector<double> globalDirector(3, 0); 

        

	for (int stepCount = 0; stepCount < steps; stepCount++) {
		if (((this->timeCounter) == 0) && trajOutputFlag) {
			this->outputTrajectory(this->trajOs);
		}

                if (local) {
                    globalDirector[0] = particle->orientVec[0] * director[0] + 
                                        particle->oriMoveDirection[0][0] * director[1] + 
                                        particle->oriMoveDirection[1][0] * director[2];
                    globalDirector[1] = particle->orientVec[1] * director[0] + 
                                        particle->oriMoveDirection[0][1] * director[1] + 
                                        particle->oriMoveDirection[1][1] * director[2];
                    globalDirector[2] = particle->orientVec[2] * director[0] + 
                                        particle->oriMoveDirection[0][2] * director[1] + 
                                        particle->oriMoveDirection[1][2] * director[2];
                } else {
                    globalDirector[0] = director[0];
                    globalDirector[1] = director[1];
                    globalDirector[2] = director[2];            
                }
                
		particle->F[2] = gravity;

		for (int i = 0; i < 3; i++) {
			particle->r[i] += (mobility * particle->F[i] + particle->u * maxSpeed * particle->orientVec[i]) * dt_;
			particle->r[i] += ambientFieldVelocity[i] * dt_;
		}

		double dot = particle->orientVec[0] * globalDirector[0] + particle->orientVec[1] * globalDirector[1] + particle->orientVec[2] * globalDirector[2];
		for (int i = 0; i < 3; i++) {
			
			particle->orientVec[i] += (globalDirector[i] - dot * particle->orientVec[i]) * dt_ * relaxationTimeInverse;
		}

		if (randomMoveFlag) {
			double randomPos[3], randomOriMove[2];

			for (int i = 0; i < 3; i++) {
				randomPos[i] = sqrt(2.0 * diffusivity_t * dt_) * rand_normal(rand_generator);
				particle->r[i] += randomPos[i];
			}
			for (int i = 0; i < 2; i++) {
				randomOriMove[i] = sqrt(2.0 * diffusivity_r * dt_) * rand_normal(rand_generator);

			}
			for (int i = 0; i < 3; i++) {
				particle->orientVec[i] += (particle->oriMoveDirection[0][i] * randomOriMove[0] + particle->oriMoveDirection[1][i] * randomOriMove[1]) * dt_;
			}

		}
		particle->normalizeOri();


		this->timeCounter++;
		if (((this->timeCounter) % trajOutputInterval == 0) && trajOutputFlag) {
			particle->normalizeOri();
			this->outputTrajectory(this->trajOs);
		}
	}
	particle->updateAngles();
}


void ActiveParticle3DSimulator::run(int steps, const std::vector<double>& actions) {


	if (particle->type == ParticleType::VANILLASP) {
		particle->u = actions[0];
		particle->v = 0.0;
		particle->w = 0.0;
	}

	if (particle->type == ParticleType::SLIDER) {
		particle->u = 1.0;
		particle->v = actions[0];
		particle->w = actions[1];
	}


	for (int stepCount = 0; stepCount < steps; stepCount++) {
		if (((this->timeCounter) == 0) && trajOutputFlag) {
			this->outputTrajectory(this->trajOs);
		}


		/*
		oriMoveDirection[0][0] = -sin(particle->phi);
		oriMoveDirection[0][1] = cos(particle->phi);
		oriMoveDirection[0][2] = 0.0;

		oriMoveDirection[1][0] = (particle->orientVec[1] * oriMoveDirection[0][2] - particle->orientVec[2] * oriMoveDirection[0][1]);
		oriMoveDirection[1][1] = (particle->orientVec[2] * oriMoveDirection[0][0] - particle->orientVec[0] * oriMoveDirection[0][2]);
		oriMoveDirection[1][2] = (particle->orientVec[0] * oriMoveDirection[0][1] - particle->orientVec[1] * oriMoveDirection[0][0]);
		*/
		particle->F[2] = gravity;

		for (int i = 0; i < 3; i++) {
			particle->r[i] += (mobility * particle->F[i] + particle->u * maxSpeed * particle->orientVec[i]) * dt_;
			particle->r[i] += ambientFieldVelocity[i] * dt_;
		}

		for (int i = 0; i < 3; i++) {
			particle->orientVec[i] += (particle->oriMoveDirection[0][i] * maxRotationSpeed * particle->v
				+ particle->oriMoveDirection[1][i] * maxRotationSpeed * particle->w) * dt_;
		}

		if (randomMoveFlag) {
			double randomPos[3], randomOriMove[2];

			for (int i = 0; i < 3; i++) {
				randomPos[i] = sqrt(2.0 * diffusivity_t * dt_) * rand_normal(rand_generator);
				particle->r[i] += randomPos[i];
			}
			for (int i = 0; i < 2; i++) {
				randomOriMove[i] = sqrt(2.0 * diffusivity_r * dt_) * rand_normal(rand_generator);

			}
			for (int i = 0; i < 3; i++) {
				particle->orientVec[i] += (particle->oriMoveDirection[0][i] * randomOriMove[0] + particle->oriMoveDirection[1][i] * randomOriMove[1]);
			}

		}

                particle->normalizeOri();


		this->timeCounter++;
		if (((this->timeCounter) % trajOutputInterval == 0) && trajOutputFlag) {
			particle->normalizeOri();
			this->outputTrajectory(this->trajOs);
		}
	}
	particle->updateAngles();
}

py::array_t<double> ActiveParticle3DSimulator::get_particle_local_frame() {
	std::vector<double> positions(9);
	positions[0] = particle->orientVec[0];
	positions[1] = particle->orientVec[1];
	positions[2] = particle->orientVec[2];
	positions[3] = particle->oriMoveDirection[0][0];
	positions[4] = particle->oriMoveDirection[0][1];
	positions[5] = particle->oriMoveDirection[0][2];
	positions[6] = particle->oriMoveDirection[1][0];
	positions[7] = particle->oriMoveDirection[1][1];
	positions[8] = particle->oriMoveDirection[1][2];

	py::array_t<double> result(9, positions.data());

	return result;

}


void ActiveParticle3DSimulator::createInitialState(double x, double y, double z, double ori0, double ori1, double ori2) {

	particle->r[0] = x*radius;
	particle->r[1] = y*radius;
	particle->r[2] = z*radius;
	particle->orientVec[0] = ori0;
	particle->orientVec[1] = ori1;
	particle->orientVec[2] = ori2;
	particle->updateAngles();
	std::stringstream ss;
	std::cout << "model initialize at round " << fileCounter << std::endl;
	ss << this->fileCounter++;
	if (trajOs.is_open() && trajOutputFlag) trajOs.close();

	if (trajOutputFlag) {
		this->trajOs.open(filetag + "xyz_" + ss.str() + ".txt");
	}

	this->timeCounter = 0;
}

void ActiveParticle3DSimulator::setInitialState(double x, double y, double z, double ori0, double ori1, double ori2){
	particle->r[0] = x*radius;
	particle->r[1] = y*radius;
	particle->r[2] = z*radius;
	particle->orientVec[0] = ori0;
	particle->orientVec[1] = ori1;
	particle->orientVec[2] = ori2;

}


void ActiveParticle3DSimulator::close() {
	if (trajOs.is_open()) trajOs.close();
}


std::vector<double> ActiveParticle3DSimulator::steeringAction(std::vector<double> target) {

	if (particle->type == ParticleType::SLIDER) {
		for (int i = 0; i < 3; i++)
			target[i] -= particle->r[i] / radius;
		double length = sqrt(target[0] * target[0] + target[1] * target[1] + target[2] * target[2]);
		for (int i = 0; i < 3; i++)
			target[i] /= length;

		double straightDirector[3];
		double dot = 0.0;
		for (int i = 0; i < 3; i++)
			dot += particle->orientVec[i] * target[i];

		for (int i = 0; i < 3; i++)
			straightDirector[i] = (target[i] - dot * particle->orientVec[i]);

		std::vector<double> action(2, 0.0);
		for (int i = 0; i < 3; i++) {
			action[0] += straightDirector[i] * particle->oriMoveDirection[0][i];
			action[1] += straightDirector[i] * particle->oriMoveDirection[1][i];
		}
		return action;
	}
	if (particle->type == ParticleType::VANILLASP) {
		for (int i = 0; i < 3; i++)
			target[i] -= particle->r[i] / radius;
		double length = sqrt(target[0] * target[0] + target[1] * target[1] + target[2] * target[2]);
		for (int i = 0; i < 3; i++)
			target[i] /= length;

		double straightDirector[3];
		double dot = 0.0;
		for (int i = 0; i < 3; i++)
			dot += particle->orientVec[i] * target[i];
		std::vector<double> action(1, 0.0);
		if (dot > 0.5) {
			action[0] = 1.0;
		}
		return action;
	}
}

void ActiveParticle3DSimulator::outputTrajectory(std::ostream& os) {


	os << this->timeCounter << "\t";
	for (int j = 0; j < dimP; j++) {
		os << particle->r[j] / radius << "\t";
	}

	os << particle->theta << "\t";
	os << particle->phi << "\t";
	os << particle->orientVec[0] << "\t";
	os << particle->orientVec[1] << "\t";
	os << particle->orientVec[2] << "\t";
	os << particle->u << "\t";
	os << particle->v << "\t";
	os << particle->w << "\t";
	os << std::endl;


}


void ActiveParticle3DSimulator::step(int nstep, py::array_t<double>& actions) {

	auto buf = actions.request();
	double *ptr = (double *)buf.ptr;
	int size = buf.size;
	std::vector<double> actions_cpp(ptr, ptr + size);


	run(nstep, actions_cpp);
}

void ActiveParticle3DSimulator::stepGivenDirector(int nstep, py::array_t<double>& actions, bool localFlag) {

	auto buf = actions.request();
	double *ptr = (double *)buf.ptr;
	int size = buf.size;
	std::vector<double> actions_cpp(ptr, ptr + size);


	run_given_director(nstep, actions_cpp, localFlag);
}