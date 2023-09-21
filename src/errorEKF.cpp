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

#include <tracking/errorEKF.h>

errorEKF::errorEKF()
{
    // Gravity Vector
    g -> setZero();
    (*g)(2) = -9.80;
}   

void errorEKF::init()
{
    firstrun = true;
    useEuler = true;

    If->setIdentity();
    P->setZero();
    //set velocity, rotational and positional errors and gyro, acc biases
    P->block<3, 3>(0, 0).setConstant(1e-3);
    P->block<3, 3>(3, 3).setConstant(1e-3);
    P->block<3, 3>(6, 6).setConstant(1e-5);
    P->block<3, 3>(9, 9).setConstant(1e-3);
    P->block<3, 3>(12, 12).setConstant(1e-3);

    Hf->setZero();
    Hf->block<3, 3>(0, 6).setIdentity();
    Hf->block<3, 3>(3, 3).setIdentity();
    Hvf->setZero();
    Hvf->block<3, 3>(3, 3).setIdentity();
    Hv->setZero();

    Rib->setIdentity();
    x->setZero();

    z->setZero();
    zv->setZero();

    v->setZero();
    dxf->setZero();
    temp->setZero();
    Kf->setZero();
    Kv->setZero();

    s->setZero();

    R->setZero();

    Acf->setZero();
    Qff->setZero();
    Qf->setZero();
    Af->setZero();

    bgyr->setZero();
    bacc->setZero();
    gyro->setZero();
    acc->setZero();
    angle->setZero();

    fhat->setZero();
    omegahat->setZero();
    v -> setZero();
    Lcf->setZero();
    Lcf->block<3, 3>(0, 3).setIdentity();
    Lcf->block<3, 3>(3, 0).setIdentity();
    Lcf->block<3, 3>(9, 6).setIdentity();
    Lcf->block<3, 3>(12, 9).setIdentity();
    r-> setZero();
    angleX = 0.0;
    angleY = 0.0;
    angleZ = 0.0;
    gyroX = 0.0;
    gyroY = 0.0;
    gyroZ = 0.0;
    accX = 0.0;
    accY = 0.0;
    accZ = 0.0;
    rX = 0.0;
    rY = 0.0;
    rZ = 0.0;
    velX = 0.0;
    velY = 0.0;
    velZ = 0.0;
    Tib->setIdentity();

    std::cout << "Base EKF Initialized Successfully" << std::endl;
}


std::unique_ptr<Eigen::Matrix<double, 15, 15>> errorEKF::computeTrans(const std::unique_ptr<Eigen::Matrix<double, 15, 1>>& x_,
                                                                    const std::unique_ptr<Eigen::Matrix<double, 3, 3>>& Rib_,
                                                                    const std::unique_ptr<Eigen::Vector3d>& omega_,
                                                                    const std::unique_ptr<Eigen::Vector3d>& f_)
{
    (*omega_).noalias() -= (*x_).segment<3>(9);
    (*f_).noalias() -= (*x_).segment<3>(12);
    v->segment<3>(0) = (*x_).segment<3>(0);
    std::unique_ptr<Eigen::Matrix<double, 15, 15>> res = std::make_unique<Eigen::Matrix<double, 15, 15>>();
    (*res).setZero();
    (*res).block<3, 3>(0, 0).noalias() = -(*wedge(omega_));
    auto j = (Rib_-> transpose());
    Vector3d k = j*(*g); 
    const std::unique_ptr<Eigen::Vector3d> a;
    (*a) = k;
    (*res).block<3, 3>(0, 3).noalias() = (*wedge(a));
    (*res).block<3, 3>(0, 12).noalias() = -Eigen::Matrix3d::Identity();
    (*res).block<3, 3>(0, 9).noalias() = -(*wedge(v));
    (*res).block<3, 3>(3, 3).noalias() = -(*wedge(omega_));
    (*res).block<3, 3>(3, 9).noalias() = -Eigen::Matrix3d::Identity();
    (*res).block<3, 3>(6, 0) = *Rib_;
    (*res).block<3, 3>(6, 3).noalias() = -(*Rib_) * (*wedge(v));
    
    return res;
}


void errorEKF::euler(const std::unique_ptr<Vector3d>& omega_,const std::unique_ptr<Vector3d>& f_)
{
    Acf = computeTrans(x, Rib, omega_, f_);
    // Euler Discretization - First order Truncation
    (*Af) = (*If);
    Af->noalias() += (*Acf) * dt;
    x = computeDiscreteDyn(x, Rib, omega_, f_);
    // x.noalias() += computeContinuousDyn(x,Rib,omega_,f_)*dt;
}

std::unique_ptr<Matrix<double, 15, 1>> errorEKF::computeDiscreteDyn(const std::unique_ptr<Matrix<double, 15, 1>>& x_,
														  const std::unique_ptr<Eigen::Matrix3d>& Rib_,
														  const std::unique_ptr<Vector3d>& omega_,
														  const std::unique_ptr<Vector3d>& f_)
{
    std::unique_ptr<Matrix<double, 15, 1>> res = std::make_unique<Eigen::Matrix<double, 15, 1>>();

    omega_->noalias() -= x_->segment<3>(9);
    f_->noalias() -= x_->segment<3>(12);
    v->segment<3>(0) = x_->segment<3>(0);
    res->segment<3>(0).noalias() = v->cross((*omega_));
    res->segment<3>(0).noalias() += Rib_->transpose() * (*g);
    res->segment<3>(0) += (*f_);

    // Position
    (*r) = x_->segment<3>(6);
    res->segment<3>(6).noalias() = (*Rib_)* res->segment<3>(0) * dt * dt / 2.00;
    res->segment<3>(6).noalias() += (*Rib_) * (*v) * dt;
    res->segment<3>(6) += (*r);
    // Velocity
    res->segment<3>(0) *= dt;
    res->segment<3>(0) += (*v);

    // Biases
    res->segment<3>(9) = x_->segment<3>(9);
    res->segment<3>(12) = x_->segment<3>(12);
    return res;
    // Nonlinear Process Model#include <stdio.h>


}

void errorEKF::predict(const std::unique_ptr<Vector3d>& omega_, const std::unique_ptr<Vector3d>& f_)
{

    //save new values of IMU readings (inputs)
    (*omega) = (*omega_);
    (*f) = (*f_);
    //calculate omegahat (new reading - bias)
    omegahat->noalias() = (*omega) - x->segment<3>(9);
    //save old speed 
    (*v) = x->segment<3>(0);

    // Update the Input-noise Jacobian
    Lcf->block<3, 3>(0, 0).noalias() = -(*wedge(v));

    euler(omega_, f_);

    // Covariance Q with full state + biases
    (*Qf)(0, 0) = gyr_qx * gyr_qx;
    (*Qf)(1, 1) = gyr_qy * gyr_qy;
    (*Qf)(2, 2) = gyr_qz * gyr_qz;
    (*Qf)(3, 3) = acc_qx * acc_qx;
    (*Qf)(4, 4) = acc_qy * acc_qy;
    (*Qf)(5, 5) = acc_qz * acc_qz;
    (*Qf)(6, 6) = gyrb_qx * gyrb_qx;
    (*Qf)(7, 7) = gyrb_qy * gyrb_qy;
    (*Qf)(8, 8) = gyrb_qz * gyrb_qz;
    (*Qf)(9, 9) = accb_qx * accb_qx;
    (*Qf)(10, 10) = accb_qy * accb_qy;
    (*Qf)(11, 11) = accb_qz * accb_qz;

    Qff->noalias() = (*Af) * (*Lcf) * (*Qf) * Lcf->transpose() * Af->transpose() * dt;
    /** Predict Step: Propagate the Error Covariance  **/
    P->noalias() = (*Af) * (*P) * Af->transpose() + (*Qff);

    // Propagate only if non-zero input
    //update rotation separately
    if (!omegahat->isZero())
    {   
        std::unique_ptr<Vector3d> temp;
        (*temp) = (*omegahat) * dt;
        //update Rib
        (*Rib) *= (*expMap(temp));
    }

    x->segment<3>(3).setZero();
    //update x state
    updateVars();
}

void errorEKF::updateWithTwistRotation(const std::unique_ptr<Vector3d>& y,const std::unique_ptr<Eigen::Quaterniond>& qy)
{

    (*R)(0, 0) = vel_px * vel_px;
    (*R)(1, 1) = vel_py * vel_py;
    (*R)(2, 2) = vel_pz * vel_pz;
    (*R)(3, 3) = leg_odom_ax * leg_odom_ax;
    (*R)(4, 4) = leg_odom_ay * leg_odom_ay;
    (*R)(5, 5) = leg_odom_az * leg_odom_az;

    (*v) = x->segment<3>(0);
    // std::cout<<" Update with Twist Rot" <<std::endl;
    // std::cout<<y<<std::endl;
    // Innovetion vector
    z->segment<3>(0) = (*y);
    z->segment<3>(0).noalias() -= (*Rib) * (*v);    
    std::unique_ptr<Eigen::Matrix<double, 3, 3>> temp;
    (*temp) = Rib->transpose() * qy->toRotationMatrix();
    z->segment<3>(3) = (*logMap(temp));
    // z.segment<3>(3) = logMap((qy.toRotationMatrix() * Rib.transpose() ));

    Hvf->block<3, 3>(0, 0) = (*Rib);
    Hvf->block<3, 3>(0, 3).noalias() = -(*Rib) * (*wedge(v));
    (*s) = (*R);
    s->noalias() += (*Hvf) * (*P) * Hvf->transpose();
    Kf->noalias() = (*P) * Hvf->transpose() * s->inverse();

    dxf->noalias() = (*Kf) * (*z);

    // Update the mean estimate
    x->noalias() += (*dxf);

    // Update the error covariance
    (*P) = ((*If) - (*Kf) * (*Hvf)) * (*P) * ((*If) - Hvf->transpose() * Kf->transpose());
    P->noalias() += (*Kf) * (*R) * Kf->transpose();

    if ((*dxf)(3) != 0 || (*dxf)(4) != 0 || (*dxf)(5) != 0)
    {   
        const std::unique_ptr<Vector3d> temp;
        (*temp) = dxf->segment<3>(3);
        //update Rib
        (*Rib) *= (*expMap(temp));
    }
    x->segment<3>(3) = Vector3d::Zero();

    updateVars();
}

void errorEKF::updateWithLegOdom(const std::unique_ptr<Vector3d>& y,const std::unique_ptr<Eigen::Quaterniond>& qy)
{
    (*R)(0, 0) = leg_odom_px * leg_odom_px;
    (*R)(1, 1) = leg_odom_py * leg_odom_py;
    (*R)(2, 2) = leg_odom_pz * leg_odom_pz;

    (*R)(3, 3) = leg_odom_ax * leg_odom_ax;
    (*R)(4, 4) = leg_odom_ay * leg_odom_ay;
    (*R)(5, 5) = leg_odom_az * leg_odom_az;

    (*r) = x->segment<3>(6);

    // Innovetion vector
    z->segment<3>(0) = (*y) - (*r);
    std::unique_ptr<Eigen::Matrix3d> temp;
    (*temp) = (Rib->transpose() * qy->toRotationMatrix());
    z->segment<3>(3) = (*logMap(temp));

    // Compute the Kalman Gain
    (*s) = (*R);
    s->noalias() += (*Hf) * (*P) * Hf->transpose();
    Kf->noalias() = (*P) * Hf->transpose() * s->inverse();

    // Update the error covariance
    (*P) = ((*If) - (*Kf) * (*Hf)) * (*P) * ((*If) - (*Kf) * (*Hf)).transpose();
    P->noalias() += (*Kf) * (*R) * Kf->transpose();

    dxf->noalias() = (*Kf) * (*z);
    x->noalias() += (*dxf);
    if ((*dxf)(3) != 0 || (*dxf)(4) != 0 || (*dxf)(5) != 0)
    {
        std::unique_ptr<Eigen::Vector3d> temp;
        (*temp) = dxf->segment<3>(3);
        //update Rib
        (*Rib) *= (*expMap(temp));
    }
    x->segment<3>(3) = Vector3d::Zero();

    updateVars();
}

void errorEKF::updateVars()
{

    (*pos) = x->segment<3>(6);
    rX = (*pos)(0);
    rY = (*pos)(1);
    rZ = (*pos)(2);
    Tib->linear() = (*Rib);
    Tib->translation() = (*pos);
    (*qib) = Quaterniond(Tib->linear());

    // Update the biases
    (*bgyr) = x->segment<3>(9);
    (*bacc) = x->segment<3>(12);
    bias_gx = (*x)(9);
    bias_gy = (*x)(10);
    bias_gz = (*x)(11);
    bias_ax = (*x)(12);
    bias_ay = (*x)(13);
    bias_az = (*x)(14);

    (*omegahat) = (*omega) - (*bgyr);
    (*fhat) = (*f) - (*bacc);

    (*gyro) = (*Rib) * (*omegahat);
    gyroX = (*gyro)(0);
    gyroY = (*gyro)(1);
    gyroZ = (*gyro)(2);

    (*acc) = (*Rib) * (*fhat);
    accX = (*acc)(0);
    accY = (*acc)(1);
    accZ = (*acc)(2);

    (*v) = x->segment<3>(0);
    (*vel) = (*Rib) * (*v);
    velX = (*vel)(0);
    velY = (*vel)(1);
    velZ = (*vel)(2);

    // ROLL - PITCH - YAW
    angle = getEulerAngles(Rib);
    angleX = (*angle)(0);
    angleY = (*angle)(1);
    angleZ = (*angle)(2);
}

