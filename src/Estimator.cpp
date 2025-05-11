// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
std::vector<int> time_seq;
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI(10000, 1));
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI(10000, 1));
PointCovVector pv_list;
std::vector<PointPlane> ptpl_list;
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points; 
std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
bool   point_selected_surf[100000] = {0};
std::vector<M3D> crossmat_list;
int effct_feat_num = 0;
int k = 0;
int idx = -1;
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;
input_ikfom input_in;
V3D angvel_avr, acc_avr, acc_avr_norm;
int feats_down_size = 0;  
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
double G_m_s2 = 9.81;

Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_input>::type cov = MTK::get_cov<process_noise_input>::type::Zero();
	// MTK::setDiagonal<process_noise_input, vect3, 0>(cov, &process_noise_input::ng, gyr_cov_input);// 0.03
	// MTK::setDiagonal<process_noise_input, vect3, 3>(cov, &process_noise_input::na, acc_cov_input); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_input, vect3, 6>(cov, &process_noise_input::nbg, b_gyr_cov); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_input, vect3, 9>(cov, &process_noise_input::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
	Eigen::Matrix<double, 30, 30> cov;
	cov.setZero();
	cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
	cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
	cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
	cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	return cov;
}

Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); // .normalized()
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = omega[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
	vect3 a_inertial = s.rot * s.acc; // .normalized()
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = s.omg[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(acc_); // .normalized().toRotationMatrix()
	cov.template block<3, 3>(12, 18) = -s.rot; //.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(s.acc); // .normalized().toRotationMatrix()
	cov.template block<3, 3>(12, 18) = s.rot; //.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity(); 
	return cov;
}

void h_model_input(state_input &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	int effect_num_k = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		double p_norm = p_body.norm();
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{
			auto &points_near = Nearest_Points[idx+j+1];
            //ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS); // 
			
			//std::vector<PointWithCov> points;
			PointCovVector points;
			PointWithCov point_world;
			point_world.x = point_world_j.x;
			point_world.y = point_world_j.y;
			point_world.z = point_world_j.z;
			point_world.intensity = point_world_j.intensity;
			ivox_->GetClosestPoint(point_world, points, NUM_MATCH_POINTS);
			points_near.clear();
			for (auto &pnt: points) {
				PointType pt;
				pt.x = pnt.x;
				pt.y = pnt.y;
				pt.z = pnt.z;
				pt.intensity = pnt.intensity;
				points_near.emplace_back(pt);
			}
			
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
				{
					float pd2 = fabs(pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3));
					// V3D norm_vec;
					// M3D Rpf, pf;
					// pf = crossmat_list[idx+j+1];
					// // pf << SKEW_SYM_MATRX(p_body);
					// Rpf = s.rot * pf;
					// norm_vec << pabcd(0), pabcd(1), pabcd(2);
					// double noise_state = norm_vec.transpose() * (cov_p+Rpf*cov_R*Rpf.transpose())  * norm_vec + sqrt(p_norm) * 0.001;
					// // if (p_norm > match_s * pd2 * pd2)
					// double epsilon = pd2 / sqrt(noise_state);
					// // cout << "check epsilon:" << epsilon << endl;
					// double weight = 1.0; // epsilon / sqrt(epsilon * epsilon+1);
					// if (epsilon > 1.0) 
					// {
					// 	weight = sqrt(2 * epsilon - 1) / epsilon;
					// 	pabcd(0) = weight * pabcd(0);
					// 	pabcd(1) = weight * pabcd(1);
					// 	pabcd(2) = weight * pabcd(2);
					// 	pabcd(3) = weight * pabcd(3);
					// }
					if (p_norm > match_s * pd2 * pd2)
					{
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
	//ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.M_Noise.resize(effect_num_k);
	ekfom_data.h_x.resize(effect_num_k, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	
	for (int j = 0; j < time_seq[k]; j++)
	{
		// ekfom_data.converge = false;
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			
			if (extrinsic_est_en)
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{   
				M3D point_crossmat = crossmat_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec); // transpose().normalized()
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			ekfom_data.M_Noise(m) = laser_point_cov;

			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

void h_model_output(state_output &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	ptpl_list.resize(time_seq[k]);
	int effect_num_k = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		PointWithCov &pv = pv_list[idx+j+1];
		PointPlane &ptpl = ptpl_list[j];

		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		double p_norm = p_body.norm();
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;

		ptpl.point_b = p_body;
		ptpl.point_w = p_world;
		ptpl.cov_b = pv.cov;
		ptpl.cov_w = calcWorldCov(p_body, pv.cov, kf_output);
		{
			auto &points_near = Nearest_Points[idx+j+1];
			
            //ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS); // 

			PointCovVector points;
			PointWithCov point_world;
			point_world.x = point_world_j.x;
			point_world.y = point_world_j.y;
			point_world.z = point_world_j.z;
			point_world.intensity = point_world_j.intensity;
			ivox_->GetClosestPoint(point_world, points, NUM_MATCH_POINTS);
			points_near.clear();
			for (auto &pnt: points) {
				PointType pt;
				pt.x = pnt.x;
				pt.y = pnt.y;
				pt.z = pnt.z;
				pt.intensity = pnt.intensity;
				points_near.emplace_back(pt);
			}
			
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
			{
				point_selected_surf[idx+j+1] = false;
				ptpl.selected = false;
			}
			else
			{
				point_selected_surf[idx+j+1] = false;
				ptpl.selected = false;
				
				if (esti_plane_cov(ptpl, points, plane_thr)) //(planeValid)
				{
					ptpl.is_plane = true;
					ptpl.res = ptpl.normal.dot(ptpl.point_w) + ptpl.d;

					Matrix<double, 1, 6> J_nq;
            		J_nq.block<1, 3>(0, 0) = ptpl.point_w - ptpl.center;
            		J_nq.block<1, 3>(0, 3) = -ptpl.normal;

					double sigma = J_nq * ptpl.cov_plane * J_nq.transpose() + ptpl.normal.dot(ptpl.cov_w * ptpl.normal);

					bool chk = cov_on? abs(ptpl.res) < sigma_num * sqrt(sigma) : p_norm > match_s * ptpl.res * ptpl.res;
					if (chk)
					{
						ptpl.selected = true;
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
	//ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.M_Noise.resize(effect_num_k);
	ekfom_data.h_x.resize(effect_num_k, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		// ekfom_data.converge = false;
		PointPlane &ptpl = ptpl_list[j];
		if (ptpl.selected) {
			M3D point_crossmat = crossmat_list[idx+j+1];
			V3D C(s.rot.transpose() * ptpl.normal); // transpose().normalized()
			V3D A(point_crossmat * C);
			ekfom_data.h_x.block<1, 12>(m, 0) << ptpl.normal(0), ptpl.normal(1), ptpl.normal(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			ekfom_data.z(m) = -ptpl.res;
			
			auto &s = kf_output.get_x();
			M3D R_cov_Rt = s.rot * s.offset_R_L_I * ptpl.cov_b * s.offset_R_L_I.transpose() * s.rot.transpose();
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = ptpl.point_w - ptpl.center;
            J_nq.block<1, 3>(0, 3) = -ptpl.normal;
            double sigma_l = J_nq * ptpl.cov_plane * J_nq.transpose();
			ekfom_data.M_Noise(m) = cov_on? sigma_l + ptpl.normal.transpose() * R_cov_Rt * ptpl.normal : laser_point_cov;

			m++;
		}
		
	}
	effct_feat_num += effect_num_k;
}

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    std::memset(ekfom_data.satu_check, false, 6);
	ekfom_data.z_IMU.block<3,1>(0, 0) = angvel_avr - s.omg - s.bg;
	ekfom_data.z_IMU.block<3,1>(3, 0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
	if(check_satu)
	{
		if(fabs(angvel_avr(0)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[0] = true; 
			ekfom_data.z_IMU(0) = 0.0;
		}
		
		if(fabs(angvel_avr(1)) >= 0.99 * satu_gyro) 
		{
			ekfom_data.satu_check[1] = true;
			ekfom_data.z_IMU(1) = 0.0;
		}
		
		if(fabs(angvel_avr(2)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[2] = true;
			ekfom_data.z_IMU(2) = 0.0;
		}
		
		if(fabs(acc_avr(0)) >= 0.99 * satu_acc)
		{
			ekfom_data.satu_check[3] = true;
			ekfom_data.z_IMU(3) = 0.0;
		}

		if(fabs(acc_avr(1)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[4] = true;
			ekfom_data.z_IMU(4) = 0.0;
		}

		if(fabs(acc_avr(2)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[5] = true;
			ekfom_data.z_IMU(5) = 0.0;
		}
	}
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{    
    V3D p_body(pi->x, pi->y, pi->z);
    
    V3D p_global;
	if (extrinsic_est_en)
	{	
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
		}
		else
		{
			p_global = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
		}
	}
	else
	{
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos; // .normalized()
		}
		else
		{
			p_global = kf_input.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_input.x_.pos; // .normalized()
		}
	}

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

Matrix3d calcLidarCov(const Vector3d &p_lidar, const double ranging_cov, const double angle_cov) {
	double range2 = p_lidar.squaredNorm();
    Matrix3d wwT = p_lidar * p_lidar.transpose() / range2;
    //return ranging_cov * wwT + angle_cov * range2 * (Eye3d - wwT);
	return ranging_cov * wwT + angle_cov * sqrt(range2) * (Eye3d - wwT);
	//return ranging_cov * wwT + angle_cov * (Eye3d - wwT);
}

Matrix3d calcWorldCov(const Vector3d &p_lidar, const Matrix3d &cov_lidar, esekfom::esekf<state_output, 30, input_ikfom> &kf_output) {
    Matrix3d p_lidar_x;
    p_lidar_x << SKEW_SYM_MATRX(p_lidar);
	auto &state = kf_output.get_x();

    // lidar->body
    Matrix3d cov_body = state.offset_R_L_I * cov_lidar * state.offset_R_L_I.transpose();

    // body->world
    Vector3d p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;
    Matrix3d p_body_x;
    p_body_x << SKEW_SYM_MATRX(p_body);

    Matrix3d cov_rot = kf_output.get_P().block<3, 3>(3, 3);
    Matrix3d cov_t = kf_output.get_P().block<3, 3>(0, 0);
    Matrix3d cov_world = state.rot * cov_body * state.rot.transpose() 
        			   + state.rot * p_body_x * cov_rot * p_body_x.transpose() * state.rot.transpose() 
        			   + cov_t;

    return cov_world;
}