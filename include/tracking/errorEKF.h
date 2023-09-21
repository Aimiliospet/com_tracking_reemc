/*
 * Copyright 2017-2023 Stylianos Piperakis,
 * Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas
 *       (FORTH) nor the names of its contributors may be used to endorse or
 *       promote products derived from this software without specific prior
 *       written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 /**
 * @brief Base Estimator combining Inertial Measurement Unit (IMU) and
 * Odometry Measuruements either from leg odometry or external odometry e.g
 * Visual Odometry (VO) or Lidar Odometry (LO)
 * @author Stylianos Piperakis
 * @details State is  position in World frame
 * velocity in  Base frame
 * orientation of Body frame wrt the World frame
 * accelerometer bias in Base frame
 * gyro bias in Base frame
 * Measurements are: Base Position/Orinetation in World frame by Leg Odometry
 * or Visual Odometry (VO) or Lidar Odometry (LO), when VO/LO is considered the
 * kinematically computed base velocity (Twist) is also employed for update.
 */
#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <memory>

using namespace Eigen;

class errorEKF {

private: 
	std::unique_ptr<Matrix<double, 15, 12>> Lcf;
	/// Error Covariance, Linearized state transition model, Identity matrix,
	/// state uncertainty matrix
	std::unique_ptr<Matrix<double, 15, 15>> P, Af, Acf, If, Qff;
	/// Linearized Measurement model
	std::unique_ptr<Matrix<double, 6, 15>> Hf, Hvf;
	std::unique_ptr<Matrix<double, 3, 15>> Hv;
	/// State-Input Uncertainty matrix
	std::unique_ptr<Matrix<double, 12, 12>> Qf;
	/// Kalman Gain
	std::unique_ptr<Matrix<double, 15, 6>> Kf;
	std::unique_ptr<Matrix<double, 15, 3>> Kv;
	/// Correction state vector
	std::unique_ptr<Matrix<double, 15, 1>> dxf;
	/// Update error covariance and Measurement noise
	std::unique_ptr<Matrix<double, 6, 6>> s, R;
	/// position, velocity , acc bias, gyro bias, bias corrected acc, bias corrected gyr, temp vectors
	std::unique_ptr<Vector3d> r, v, omega, f, fhat, omegahat, temp;
	/// Innovation vectors
	std::unique_ptr<Matrix<double, 6, 1>> z;
	std::unique_ptr<Vector3d> zv;


	/**
	 * @brief computes the state transition matrix for linearized error state dynamics
	 *
	 */
	std::unique_ptr<Matrix<double, 15, 15>> computeTrans(const std::unique_ptr<Matrix<double, 15, 1>>& x_,
														const std::unique_ptr<Eigen::Matrix3d>& Rib_,
														const std::unique_ptr<Vector3d>& omega_,
														const std::unique_ptr<Vector3d>& f_);

	/**
	 * @brief performs euler (first-order) discretization to the nonlinear state-space dynamics
	 *
	 */
	void euler(const std::unique_ptr<Vector3d>& omega_,const std::unique_ptr<Vector3d>& f_);

	/**
	 * @brief computes the discrete-time nonlinear state-space dynamics
	 *
	 */
	std::unique_ptr<Matrix<double, 15, 1>> computeDiscreteDyn(const std::unique_ptr<Matrix<double, 15, 1>>& x_,
															const std::unique_ptr<Eigen::Matrix3d>& Rib_,
															const std::unique_ptr<Vector3d>& omega_,
															const std::unique_ptr<Vector3d>& f_);

public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        std::unique_ptr<Matrix<double, 15, 1>> x;
        bool firstrun;
		bool useEuler;
        std::unique_ptr<Vector3d> g, bgyr, bacc, gyro, acc, vel, pos, angle;
        double acc_qx, acc_qy, acc_qz, gyr_qx, gyr_qy, gyr_qz, gyrb_qx, gyrb_qy, gyrb_qz,
				accb_qx, accb_qy, accb_qz, odom_px, odom_py, odom_pz, odom_ax, odom_ay, odom_az,
				vel_px, vel_py, vel_pz, leg_odom_px, leg_odom_py, leg_odom_pz, leg_odom_ax,
				leg_odom_ay, leg_odom_az;

		double gyroX, gyroY, gyroZ, angleX, angleY, angleZ, bias_gx, bias_gy, bias_gz,
				bias_ax, bias_ay, bias_az, ghat;

		double accX, accY, accZ, velX, velY, velZ, rX, rY, rZ;
        std::unique_ptr<Eigen::Matrix3d> Rib;
        std::unique_ptr<Eigen::Affine3d> Tib;
        std::unique_ptr<Eigen::Quaterniond> qib;
        double dt;
        errorEKF();
        /** @fn void setdt(double dtt)
		 *  @brief sets the discretization of the Error State Kalman Filter (ESKF)
		 *  @param dtt sampling time in seconds
		 */
		void updateVars();
		/** @fn void setdt(double dtt)
		 *  @brief sets the discretization of the Error State Kalman Filter (ESKF)
		 *  @param dtt sampling time in seconds
		 */
		void setdt(double dtt)
		{
			dt = dtt;
		}
		/** @fn void setGyroBias(Vector3d bgyr_)
		 *  @brief initializes the angular velocity bias state of the Error State Kalman Filter (ESKF)
		 *  @param bgyr_ angular velocity bias in the base coordinates
		 */
		/*
		void setGyroBias(std::unique_ptr<Vector3d> bgyr_)
		{

            *bgyr = *bgyr_;
            x->segment<3>(9) = *bgyr;
			bias_gx = (*bgyr_)(0);
			bias_gy = (*bgyr_)(1);
			bias_gz = (*bgyr_)(2);
		}
		*/
		void setGyroBias(Vector3d bgyr_)
		{

            *bgyr = bgyr_;
            x->segment<3>(9) = *bgyr;
			bias_gx = bgyr_(0);
			bias_gy = bgyr_(1);
			bias_gz = bgyr_(2);
		}
        /** @fn void setAccBias(Vector3d bacc_)
		 *  @brief initializes the acceleration bias state of the Error State Kalman Filter (ESKF)
		 *  @param bacc_ acceleration bias in the base coordinates
		 */
		/*void setAccBias(std::unique_ptr<Vector3d> bacc_)
		{
			*bacc = *bacc_;
			x->segment<3>(12) = *bacc;
			bias_ax = (*bacc)(0);
			bias_ay = (*bacc)(1);
			bias_az = (*bacc)(2);
		}*/
		void setAccBias(Vector3d bacc_)
		{
			*bacc = bacc_;
			x->segment<3>(12) = *bacc;
			bias_ax = bacc_(0);
			bias_ay = bacc_(1);
			bias_az = bacc_(2);
		}
        /** @fn void setBodyPos(Vector3d bp)
		 *  @brief initializes the base position state of the Error State Kalman Filter (ESKF)
		 *  @param bp Position of the base in the world frame
		 */
		/*void setBodyPos(std::unique_ptr<Vector3d> bp)
		{
			x->segment<3>(6) = *bp;
		}*/
		void setBodyPos(Vector3d bp)
		{
			x->segment<3>(6) = bp;
		}

        /** @fn void setBodyOrientation(Matrix3d Rot_)
		 *  @brief initializes the base rotation state of the Error State Kalman Filter (ESKF)
		 *  @param Rot_ Rotation of the base in the world frame
		 */
		/*void setBodyOrientation(std::unique_ptr<Eigen::Matrix3d> Rot_)
		{
			*Rib = *Rot_;
		}*/
		void setBodyOrientation(Eigen::Matrix3d Rot_)
		{
			*Rib = Rot_;
		}

        /** @fn void setBodyVel(Vector3d bv)
		 *  @brief initializes the base velocity state of the Error State Kalman Filter (ESKF)
		 *  @param bv linear velocity of the base in the base frame
		 */
		/*void setBodyVel(std::unique_ptr<Vector3d> bv)
		{
			x->segment<3>(0).noalias() = *bv;
		}*/
		void setBodyVel(Vector3d bv)
		{
			x->segment<3>(0).noalias() = bv;
		}

        /** @fn void predict(Vector3d omega_, Vector3d f_);
		 *  @brief realises the predict step of the Error State Kalman Filter (ESKF)
		 *  @param omega_ angular velocity of the base in the base frame
		 *  @param f_ linear acceleration of the base in the base frame
		 */
		void predict(const std::unique_ptr<Vector3d>& omega_, const std::unique_ptr<Vector3d>& f_);


		/** @fn void updateWithLegOdom(Vector3d y, Quaterniond qy);
		 *  @brief realises the pose update step of the Error State Kalman Filter (ESKF) with Leg Odometry
		 *  @param y 3D base position measurement in the world frame
		 *  @param qy orientation of the base w.r.t the world frame in quaternion
		 *  @note Leg odometry is accurate when accurate contact states are detected
		 */
		void updateWithLegOdom(const std::unique_ptr<Vector3d>& y,const std::unique_ptr<Eigen::Quaterniond>& qy);
		
		/** @fn void updateWithTwistRotation(Vector3d y,Quaterniond qy);
		 *  @brief realises the  update step of the Error State Kalman Filter (ESKF) with a base linear velocity measurement and orientation measurement
		 *  @param y 3D base velociy measurement in the world frame
		 * 	@param qy orientation of the base w.r.t the world frame in quaternion
		 */
		void updateWithTwistRotation(const std::unique_ptr<Vector3d>& y,const std::unique_ptr<Eigen::Quaterniond>& qy);
		/**
		 *  @fn void init()
		 *  @brief Initializes the Base Estimator
		 *  @details
		 *   Initializes:  State-Error Covariance  P, State x, Linearization Matrices for process and measurement models Acf, Lcf, Hf and rest class variables
		 */
		void init();
		/** @fn Matrix3d wedge(Vector3d v)
		 * 	@brief Computes the skew symmetric matrix of a 3-D vector
		 *  @param v  3D Twist vector
		 *  @return   3x3 skew symmetric representation
         */
        std::unique_ptr<Eigen::Matrix3d> wedge(const std::unique_ptr<Vector3d>& v) {
            
			std::unique_ptr<Eigen::Matrix3d> skew = std::make_unique<Eigen::Matrix3d>();

            (*skew) << 0, -(*v)(2), (*v)(1),
                    (*v)(2), 0, -(*v)(0),
                    -(*v)(1), (*v)(0), 0;

             return skew;
        }
        /** @fn Vector3d vec(Matrix3d M)
		 *  @brief Computes the vector represation of a skew symmetric matrix
		 *  @param M  3x3 skew symmetric matrix
		 *  @return   3D Twist vector
		 */
        std::unique_ptr<Vector3d> vec(std::unique_ptr<Eigen::Matrix3d>& M) {
            
			std::unique_ptr<Vector3d> v = std::make_unique<Vector3d>();

            (*v) << (*M)(2, 1), (*M)(0, 2), (*M)(1, 0);

            return v;
        }

        /** @brief Computes the exponential map according to the Rodriquez Formula for component in so(3)
		 *  @param omega 3D twist in so(3) algebra
		 *  @return   3x3 Rotation in  SO(3) group
		 */
		inline std::unique_ptr<Eigen::Matrix<double, 3, 3>> expMap(
				const std::unique_ptr<Vector3d>& omega)
		{

			std::unique_ptr<Eigen::Matrix<double, 3, 3>> res = std::make_unique<Eigen::Matrix<double, 3, 3>>();
			double omeganorm = omega -> norm();
			res->setIdentity();

			if (omeganorm > std::numeric_limits<double>::epsilon())
			{
			    std::unique_ptr<Eigen::Matrix3d> omega_skew = wedge(omega);
				(*res) += (*omega_skew) * (sin(omeganorm) / omeganorm);
				(*res) += ((*omega_skew) * (*omega_skew)) * ((1.000 - cos(omeganorm)) / (omeganorm * omeganorm));
			}

			return res;
		}
        
        /** @brief Computes the logarithmic map for a component in SO(3) group
		 *  @param Rt 3x3 Rotation in SO(3) group
		 *  @return   3D twist in so(3) algebra
		 */
		inline std::unique_ptr<Vector3d> logMap(
				const std::unique_ptr<Eigen::Matrix<double, 3, 3>>& Rt)
		{

			std::unique_ptr<Vector3d> res = std::make_unique<Vector3d>();
			double costheta = (Rt->trace() - 1.0) / 2.0;
			double theta = acos(costheta);

			if (fabs(theta) > std::numeric_limits<double>::epsilon())
			{
				std::unique_ptr<Eigen::Matrix<double, 3, 3>> lnR = std::make_unique<Eigen::Matrix<double, 3, 3>>();
				lnR->noalias() = (*Rt) - Rt->transpose();
				*lnR *= theta / (2.0 * sin(theta));
				res = vec(lnR);
			}

			return res;
		}

        /** @brief Computes Euler Angles from a Rotation Matrix
		 *  @param Rt 3x3 Rotation in SO(3) group
		 *  @return   3D Vector with Roll-Pitch-Yaw
		 */
		inline std::unique_ptr<Vector3d> getEulerAngles(
				const std::unique_ptr<Eigen::Matrix3d>& Rt)
		{
			std::unique_ptr<Vector3d> res = std::make_unique<Vector3d>();
            (*res) << atan2((*Rt)(2, 1), (*Rt)(2, 2)),
                    atan2(-(*Rt)(2, 0), sqrt(pow((*Rt)(2, 1), 2) + pow((*Rt)(2, 2), 2))),
                    atan2((*Rt)(1, 0), (*Rt)(0, 0));
			
			return res;
		}

        /** @brief Computes Rotation Matrix from Euler Angles according to YPR convention
		 *  @param angles_ 3D Vector with Roll-Pitch-Yaw
		 *  @return  3x3 Rotation in SO(3) group
		 */
		inline std::unique_ptr<Eigen::Matrix3d> getRotationMatrix(
				std::unique_ptr<Vector3d> angles_)
		{
			auto res = std::make_unique<Eigen::Matrix3d>();

            double cosTheta = cos((*angles_)(2));
            double sinTheta = sin((*angles_)(2));
            double cosPhi = cos((*angles_)(1));
            double sinPhi = sin((*angles_)(1));
            double cosPsi = cos((*angles_)(0));
            double sinPsi = sin((*angles_)(0));

            (*res) << cosTheta * cosPhi, -sinTheta * cosPsi + cosTheta * sinPhi * sinPsi, sinTheta * sinPsi + cosTheta * sinPhi * cosPsi,
                        sinTheta * cosPhi, cosTheta * cosPsi + sinTheta * sinPhi * sinPsi, -cosTheta * sinPsi + sinTheta * sinPhi * cosPsi,
                        -sinPhi, cosPhi * sinPsi, cosPhi * cosPsi;

			return res;
		}
};