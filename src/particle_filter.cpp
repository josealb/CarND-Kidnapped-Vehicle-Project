/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	cout << "Initial vehicle position X: " << x << " Y: " << y << " theta: " << theta << endl;
	num_particles=500;

	default_random_engine gen;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> dist_x(x,std_x);
	normal_distribution<double> dist_y(y,std_y);
	normal_distribution<double> dist_theta(theta,std_theta);

	for (int i=0;i<num_particles;i++)
	{
		double sample_x, sample_y, sample_theta;
		Particle sampleParticle;
		sampleParticle.id=i;
		sampleParticle.x = dist_x(gen);
		sampleParticle.y = dist_y(gen);
		sampleParticle.theta = dist_theta(gen);
		sampleParticle.weight = 1;
		particles.push_back(sampleParticle);
	}
	for (int i=0;i<num_particles;i++){
		weights.push_back(1);
	}
//printParticles("After initialization");

is_initialized = true;
}
void ParticleFilter::printParticles(std::string label){
	cout << label << endl;
	for (int i=0;i<particles.size();i++){
		cout << "ID: " <<particles[i].id << " X: " << particles[i].x << "Y: " << particles[i].y << " theta: " << particles[i].theta <<" weight: " << particles[i].weight << endl;
	}
	cout << "------------" << endl;
}
double ParticleFilter::addGaussianNoise(double mean, double std_dev){

	std::random_device rd;
	std::mt19937 gen(rd());

	normal_distribution<double> dist(mean,std_dev);
	return dist(gen);
}
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	cout << "Delta T: " << delta_t << " Velocity: " << velocity << " Yaw Rate: " << yaw_rate << endl;
 	//printParticles("Before prediction");
	for (int i=0;i<num_particles;i++)
	{
		float x =  particles[i].x;
		float y =  particles[i].y;
		float theta =  particles[i].theta;

		float new_x, new_y, new_theta;
		new_x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
		new_y = y + (velocity / yaw_rate) * (cos(theta)-cos(theta+yaw_rate*delta_t));
		new_theta = theta + yaw_rate * delta_t;

		new_x = addGaussianNoise(new_x,std_pos[0]);
		new_y = addGaussianNoise(new_y,std_pos[1]);
		new_theta = addGaussianNoise(new_theta,std_pos[2]);

		particles[i].x = new_x;
		particles[i].y = new_y;
		particles[i].theta = new_theta;
//		cout << "After " << delta_t << "ms" << "particle is predicted" << endl << "X: " << particles[i].x << " Y: " << particles[i].y;
	}
	//printParticles("After prediction");
}

void ParticleFilter::dataAssociation(Particle& particle, std::vector<Map::single_landmark_s> landmark_list, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double smallestDistance;
	int bestMatchID;

	std::vector<double> sense_x;
	std::vector<double> sense_y;
	std::vector<int> associations;

	double currentSense_x, currentSense_y;

	for (int i=0;i<observations.size();i++){
		//cout << "Associating observation: " << std::to_string(i) << endl;

		smallestDistance = 99999999;
		//Compare each observation with the known landmarks to find the closest one
		for (int j=0;j<landmark_list.size();j++){ 
			double distance = dist(observations[i].x,observations[i].y,landmark_list[j].x_f,landmark_list[j].y_f);
			if (distance<smallestDistance){
				smallestDistance=distance;
				bestMatchID = landmark_list[j].id_i;
				currentSense_x = observations[i].x;
				currentSense_y = observations[i].y;
			}
		}	
	observations[i].id = bestMatchID;
	associations.push_back(bestMatchID);
	sense_x.push_back(currentSense_x);
	sense_y.push_back(currentSense_y);
	}
	//cout << "Setting associations" << endl;
	//SetAssociations(particle, associations, sense_x, sense_y);
	//cout << "Associations set" << endl;

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//printParticles("Before updating weights");
	for(int i=0;i<particles.size();i++){
		//cout << "Started processing particle: " << std::to_string(i) << endl;
		double theta = particles[i].theta;
		double posX = particles[i].x;
		double posY = particles[i].y;
		std::vector<LandmarkObs> transformedObservations;
		for (int j=0;j<observations.size();j++){
			//cout << "Started transforming observation: " << std::to_string(j) << endl;

			double obsX = observations[j].x;
			double obsY = observations[j].y;

			double mapX = cos(theta) * obsX - sin (theta) * obsY + posX;
			double mapY = sin(theta) * obsX + cos (theta) * obsY + posY;
			LandmarkObs currentObs;
			currentObs.x=mapX;
			currentObs.y=mapY;
			currentObs.id=observations[i].id;

			transformedObservations.push_back(currentObs);
		}
		//cout << "Generating associations" << endl;
		dataAssociation(particles[i],map_landmarks.landmark_list,transformedObservations);
		//cout << "Finished generating associations" << endl;

		//Now let's find the distance between the associated landmark and the observation
		std::vector<double> observation_probs;
		for (int j=0;j<transformedObservations.size();j++){
			int id = transformedObservations[j].id;
			double observationX = transformedObservations[j].x;
			double observationY = transformedObservations[j].y;
			double landmark_x,landmark_y;
			for (int k=0;k<map_landmarks.landmark_list.size();k++){ //Find the corresponding id from the association
				if (map_landmarks.landmark_list[k].id_i==id){
					landmark_x=map_landmarks.landmark_list[k].x_f;
					landmark_y=map_landmarks.landmark_list[k].y_f;
				}
			}
			//Now we calculate weights
			double landmark_std_x = std_landmark[0];
			double landmark_std_y = std_landmark[1];

			double multiplier = 1.0/(2*M_PI*landmark_std_x*landmark_std_y);
			double cov_x = pow(landmark_std_x, 2.0);
			double cov_y = pow(landmark_std_y, 2.0);
			double exponential = exp(-pow(observationX - landmark_x, 2.0)/(2.0*cov_x) - pow(observationY - landmark_y, 2.0)/(2.0*cov_y));
			double observation_prob = multiplier*exponential;
			observation_probs.push_back (observation_prob);
		}
		double particleProbabilty=observation_probs[0];
		for (int j=1;j<observation_probs.size();j++){
			particleProbabilty*=observation_probs[j];
		}
		particles[i].weight=particleProbabilty;
		weights[i]=particleProbabilty;
	}

//printParticles("After updating weights");
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	//printParticles("Before resampling");

	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());
	//std::map<int, int> m;
	std::vector<Particle> particles_new;

	for (int n=0;n<num_particles;++n){
		int index = d(gen);
		particles_new.push_back(particles[index]);
	}
	particles=particles_new;

	for (int i=0;i<particles.size();i++){
		particles[i].id = i;
	}
	//printParticles("After Resampling");

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	cout << "associations: " << endl;
	particle.associations= associations;
	cout << "sense_x: " << endl;
	particle.sense_x = sense_x;
	cout << "sense_y: " << endl;
	particle.sense_y = sense_y;
	cout << "Finished" << endl;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
