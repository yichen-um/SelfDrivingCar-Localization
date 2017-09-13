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
#include "helper_functions.h"

using namespace std;

// Declare a random engine used in multiple functions
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 101;
    
    // This line creates a normal (Gaussian) distribution for x, y and theta
    normal_distribution<double> dist_x(0, std[0]);
    normal_distribution<double> dist_y(0, std[1]);
    normal_distribution<double> dist_theta(0, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = x + dist_x(gen);
        p.y = y + dist_y(gen);
        p.theta = theta + dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    // Generate distribution for process noise
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);
    
    // Upodate the particle position and heading with motion model only
    for (int i = 0; i < num_particles; i++) {
        double x0, y0, theta0, x1, y1, theta1;
        // Progress states with bycicle model
        x0 = particles[i].x;
        y0 = particles[i].y;
        theta0 = particles[i].theta;
        if (fabs(yaw_rate) < 0.00001) {
            x1 = x0 + velocity * delta_t * cos(theta0);
            y1 = y0 + velocity * delta_t * sin(theta0);
            theta1 = theta0;
        } else {
            x1 = x0 + velocity / yaw_rate * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
            y1 = y0 + velocity / yaw_rate * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
            theta1 = theta0 + yaw_rate * delta_t;
        }
        // Add noise
        x1 += noise_x(gen);
        y1 += noise_y(gen);
        theta1 += noise_theta(gen);
        // Assign states to particles
        particles[i].x = x1;
        particles[i].y = y1;
        particles[i].theta = theta1;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to implement this method and use it as a helper during the updateWeights phase.
    
    for (int i = 0; i < observations.size(); i++) {
        // Grab current observation
        LandmarkObs curOb = observations[i];
        // Set min dist
        double minDist = numeric_limits<double>::max();
        int minId = -1;
        // Sweep each predicted (in ranged) landmarks, and find the matched observation marks
        for (int j = 0; j < predicted.size(); j++) {
            LandmarkObs curPred = predicted[j];
            double curDist = dist(curOb.x, curOb.y, curPred.x, curPred.y);
            if (curDist < minDist) {
                minDist = curDist;
                minId = curPred.id;
            }
        }
        // Associate observation ID
        observations[i].id = minId;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located according to the MAP'S coordinate system. You will need to transform between the two systems.
	//  Keep in mind that this transformation requires both rotation AND translation (but no scaling). The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    // For each particle
    for (int i = 0; i < num_particles; i++) {
        
        double pX = particles[i].x;
        double pY = particles[i].y;
        double pTheta = particles[i].theta;
        
        // Select landmarks in sensor range for each particle
        vector<LandmarkObs> landmarksInRange;
        
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            float lmX = map_landmarks.landmark_list[j].x_f;
            float lmY = map_landmarks.landmark_list[j].y_f;
            int lmId = map_landmarks.landmark_list[j].id_i;
            if (dist(pX, pY, lmX, lmY) < sensor_range) {
                landmarksInRange.push_back(LandmarkObs{ lmId, lmX, lmY });
            }
        }
        // Sweep observed landmarks, convert landmark from vehicle coordinates  (based on current particle) to global coordinates
        vector<LandmarkObs> obsMapCoord;
        
        for (int j = 0; j < observations.size(); j++) {
            int oId = observations[j].id;
            double oX = observations[j].x;
            double oY = observations[j].y;
            double tranX = pX + cos(pTheta) * oX - sin(pTheta) * oY;
            double tranY = pY + sin(pTheta) * oX + cos(pTheta) * oY;
            obsMapCoord.push_back(LandmarkObs{ oId, tranX, tranY });
        }
        
        // Associate observations with landmarks in range.
        dataAssociation(landmarksInRange, obsMapCoord);
        
        // Caculate P(x,y) of observation is the associated landmark and update weights of each particle
        double pWeight = 1.0;

        for (int j = 0; j < obsMapCoord.size(); j++) {
            double curX, curY, muX, muY;
            double stdX = std_landmark[0];
            double stdY = std_landmark[1];
            curX = obsMapCoord[j].x;
            curY = obsMapCoord[j].y;
            int curID = obsMapCoord[j].id;
            
            for (int k = 0; k < landmarksInRange.size(); k++) {
                if (landmarksInRange[k].id == curID) {
                    muX = landmarksInRange[k].x;
                    muY = landmarksInRange[k].y;
                }
            }
            
            double prob_num = exp(-( pow(curX - muX, 2) / (2 * pow(stdX, 2)) + pow(curY - muY, 2) / (2 * pow(stdY, 2)) ));
            double prob_den = 2 * M_PI * stdX * stdY;
            double prob = prob_num / prob_den;
            // Calculate the weights of corresponding particle.
            pWeight *= prob;
        }
        particles[i].weight = pWeight;
    }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
//    
    // Create a new vector for resampled particles with replacement
    vector<Particle> newParticles;
    
    // Retrive current
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    // Generate random starting index
    uniform_int_distribution<int> uniintdist(0, num_particles - 1);
    auto idx = uniintdist(gen);
    
    // Find the max weight and generate a uniform distribution
    double weightMax = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> unirealdist(0.0, weightMax);
    
    // Spin the resampling wheel
    double beta = 0.0;
    
    for (int i = 0; i < num_particles; i++) {
        beta += unirealdist(gen) * 2.0;
        while (beta > weights[idx]) {
            beta -= weights[idx];
            idx = (idx + 1) % num_particles;
        }
        newParticles.push_back(particles[idx]);
    }
    particles = newParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
