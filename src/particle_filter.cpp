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
	num_particles=100;

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
		sampleParticle.x = dist_x(gen);
		sampleParticle.y = dist_y(gen);
		sampleParticle.theta = dist_theta(gen);
		sampleParticle.weight = 1;
		particles.push_back(sampleParticle);
	}

}
double ParticleFilter::addGaussianNoise(double mean, double std_dev){
		default_random_engine gen;
		normal_distribution<double> dist(mean,std_dev);
		return dist(gen);
}
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	for (int i=0;i<num_particles;i++)
	{
		float x =  particles[i].x;
		float y =  particles[i].y;
		float theta =  particles[i].theta;

		float new_x, new_y, new_theta;
		new_x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
		new_y = y + (velocity / yaw_rate) * (cos(theta)-cos(theta-yaw_rate*delta_t));
		new_theta = theta + yaw_rate * delta_t;

		new_x = addGaussianNoise(new_x,std_pos[0]);
		new_y = addGaussianNoise(new_y,std_pos[1]);
		new_theta = addGaussianNoise(new_theta,std_pos[2]);

		particles[i].x = new_x;
		particles[i].y = new_y;
		particles[i].theta = new_theta;
	}
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> landmark_list, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double smallestDistance;
	int bestMatchID;
	for (int i=0;i<observations.size();i++){
		smallestDistance = 99999999;
		for int j=0;j<landmark_list.size();j++{
			double distance = dist(observations.x,observations.y,landmark_list[i].x,landmark_list[i].y);
			if (distance<smallestDistance){
				smallestDistance=distance;
				bestMatchID = landmark_list[i].id;
			}
		}	
	observations[i].id = bestMatchID;
	}
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
	for(int i=0;i<particles.size();i++){
		double theta = particles[i].theta;
		double posX = particles[i].x;
		double posY = particles[i].y;
		std::vector<LandmarkObs> transformedObservations;
		for (int j=0;j<observations.size();j++){
			double obsX = observations[i].x;
			double obsY = observations[i].y;

			double mapX = cos(theta) * obsX - sin (theta) * obsY + posX;
			double mapY = sin(theta) * obsY + cos (theta) * obsY + posY;
			LandmarkObs currentObs;
			currentObs.x=mapX;
			currentObs.y=mapY;
			currentObs.id=observations[i].id;

			transformedObservations.push_back(currentObs);
		}
		dataAssociation(map_landmarks.landmark_list,transformedObservations);
	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
