#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include <queue>

using namespace std;
using namespace Eigen;

struct PointWithCov {
    float x;
    float y;
    float z;
    float intensity;
    Matrix3d cov;

    Vector3f getVector3fMap() const {
        return {x, y, z};
    }
};

typedef vector<PointWithCov, Eigen::aligned_allocator<PointWithCov>>  PointCovVector;

struct PointPlane {  // point & plane pair
    bool is_plane = false;
    bool selected = false;

    Vector3d point_b;
    Vector3d point_w;
    Matrix3d cov_b;
    Matrix3d cov_w;

    Vector3d normal;
    double d;
    double res;
    Vector3d center;
    Matrix<double, 6, 6> cov_plane;
};

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((vect3, gravity))
);

MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omg))
((vect3, acc))
((vect3, gravity))
((vect3, bg))
((vect3, ba))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_input,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK_BUILD_MANIFOLD(process_noise_output,
((vect3, vel))
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

extern esekfom::esekf<state_input, 24, input_ikfom> kf_input;
extern esekfom::esekf<state_output, 30, input_ikfom> kf_output;

#define PBWIDTH 30
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

#define PI_M (3.14159265358)
// #define G_m_s2 (9.81)         // Gravaty const in GuangDong/China
#define DIM_STATE (24)      // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12)      // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN  (6.0)
#define LIDAR_SP_LEN    (2)
#define INIT_COV   (0.0001)
#define NUM_MATCH_POINTS    (5)
#define MAX_MEAS_DIM        (10000)

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define VEC_FROM_ARRAY_SIX(v)        v[0],v[1],v[2],v[3],v[4],v[5]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZRGB     PointTypeRGB;
typedef pcl::PointCloud<PointType>    PointCloudXYZI;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

#define MD(a,b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a,b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>

const M3D Eye3d(M3D::Identity());
const M3F Eye3f(M3F::Identity());
const V3D Zero3d(0, 0, 0);
const V3F Zero3f(0, 0, 0);

struct MeasureGroup     // Lidar data and imu dates for the curent process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        lidar_last_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_last_time;
    PointCloudXYZI::Ptr lidar;
    deque<sensor_msgs::Imu::ConstPtr> imu;
};

template <typename T>
T calc_dist(PointType p1, PointType p2){
    T d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

template <typename T>
T calc_dist(Eigen::Vector3d p1, PointType p2){
    T d = (p1(0) - p2.x) * (p1(0) - p2.x) + (p1(1) - p2.y) * (p1(1) - p2.y) + (p1(2) - p2.z) * (p1(2) - p2.z);
    return d;
}

template<typename T>
std::vector<int> time_compressing(const PointCloudXYZI::Ptr &point_cloud)
{
  int points_size = point_cloud->points.size();
  int j = 0;
  std::vector<int> time_seq;
  // time_seq.clear();
  time_seq.reserve(points_size);
  for(int i = 0; i < points_size - 1; i++)
  {
    j++;
    if (point_cloud->points[i+1].curvature > point_cloud->points[i].curvature)
    {
      time_seq.emplace_back(j);
      j = 0;
    }
  }
//   if (j == 0)
//   {
//     time_seq.emplace_back(1);
//   }
//   else
  {
    time_seq.emplace_back(j+1);
  }
  return time_seq;
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/
template<typename T>
bool esti_normvector(Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num)
{
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);
    
    for (int j = 0; j < point_num; j++)
    {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
        {
            return false;
        }
    }

    normvec.normalize();
    return true;
}

template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

/*
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T threshold) {
    Matrix3f H;
    Vector3f r;
    H.setZero();
    r.setZero();

    for (auto &pnt : point) {
        Vector3f p(pnt.x, pnt.y, pnt.z);
        H += p * p.transpose();
        r -= p;
    }
    
    Vector3f normvec = H.colPivHouseholderQr().solve(r);
    T n = normvec.norm();
    T lambda = (normvec.dot((H - r * r.transpose() / point.size()) * normvec)) / (n * n * point.size());

    if (lambda > threshold * threshold || abs(normvec(2)) < 1.0e-10)
        return false;

    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    return true;
}
*/

/*
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T threshold) {
    Matrix3d H;
    Vector3d r;
    H.setZero();
    r.setZero();

    for (auto &pnt : point) {
        Vector3d p(pnt.x, pnt.y, pnt.z);
        H += p * p.transpose();
        r -= p;
    }
    
    Vector3d normvec = H.colPivHouseholderQr().solve(r);
    double n = normvec.norm();
    double lambda = (normvec.dot((H - r * r.transpose() / point.size()) * normvec)) / (n * n * point.size());

    if (lambda > threshold * threshold || abs(normvec(2)) < 1.0e-10)
        return false;

    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    return true;
}
*/

/*
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const double threshold) {
    Matrix3f H;
    Vector3f r;
    H.setZero();
    r.setZero();

    for (auto &pnt : point) {
        Vector3f p(pnt.x, pnt.y, pnt.z);
        H += p * p.transpose();
        r -= p;
    }
    
    Vector3f normvec = H.colPivHouseholderQr().solve(r);
    T n = normvec.norm();
    T lambda = (normvec.dot((H - r * r.transpose() / point.size()) * normvec)) / (n * n * point.size());

    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;
    
    Matrix3d Hd;
    Vector3d rd;
    Hd.setZero();
    rd.setZero();

    for (auto &pnt : point) {
        Vector3d p(pnt.x, pnt.y, pnt.z);
        Hd += p * p.transpose();
        rd -= p;
    }
    
    Vector3d normvecd = Hd.colPivHouseholderQr().solve(rd);
    double nd = normvecd.norm();
    double lambdad = (normvecd.dot((Hd - rd * rd.transpose() / point.size()) * normvecd)) / (nd * nd * point.size());

    Vector4d pca_resultd;
    pca_resultd(0) = normvecd(0) / nd;
    pca_resultd(1) = normvecd(1) / nd;
    pca_resultd(2) = normvecd(2) / nd;
    pca_resultd(3) = 1.0 / nd;

    bool chk = (lambda > threshold * threshold);
    bool chkd = (lambdad > threshold * threshold);

    static int cnt = 0;
    static int num = 0;
    static int line = 0;
    static float avg_dist = 0;
    static Vector3d avg_norm(Vector3d::Zero());

    Vector3d center = -rd / point.size();
    Matrix3d C(Matrix3d::Zero());
    for (auto &pnt : point) {
        Vector3d p(pnt.x, pnt.y, pnt.z);
        C += (p - center) * (p - center).transpose();
    }
    C /= point.size();
    JacobiSVD<MatrixXd> svd(C, ComputeFullV | ComputeFullU);
    Vector3d d = svd.singularValues();

    cnt ++;
    
    
    if (abs(pca_result(2)) < 1.0e-20) {
        avg_dist = (avg_dist * num + center.norm()) / (num + 1); 
        avg_norm = (avg_norm * num + normvecd / nd) / (num + 1);
        num ++; 
        cout << "============== float - double difference occur !! ==============" << endl;
        
        if (d(0) > 100 * d(1)) line ++;
        cout << " is it line?" << (d(0) > 100 * d(1)) << endl;
        cout << " number of count: " << cnt << endl;
        cout << " number: " << num << endl;
        cout << " number of line: " << line << endl;
        cout << " num rate: " << (double) num / (double) cnt << endl;
        cout << " line rate: " << (double) line / (double) num << endl;
        cout << " size of C: " << C.trace() << endl;
        cout << " EigenValues: " << d.transpose() << endl;
        cout << " avg dist: " << avg_dist << endl;
        cout << " avg norm: " << avg_norm.transpose() << endl;
        cout << "---- float ----" << endl;
        cout << "  pca_result: " << pca_result.transpose() << endl;
        cout << "  center: " << center.transpose() << endl;
        cout << "  lambda: " << lambda << endl;
        cout << "---- double ----" << endl;
        cout << "  pca_resultd: " << pca_resultd.transpose() << endl;
        cout << "  center: " << center.transpose() << endl;
        cout << "  lambdad: " << lambdad << endl;
    }
    

    if (lambda > threshold * threshold) return false;
    //if (0.1 * d(1) < d(2)) return false;

    return true;
}
*/

/*
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const double threshold) {
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }
    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
    T n = normvec.norm();
    T lambda = -b.dot(A * normvec - b) / (n * n * point.size());

    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;
    
    Matrix<double, NUM_MATCH_POINTS, 3> Ad;
    Matrix<double, NUM_MATCH_POINTS, 1> bd;
    Ad.setZero();
    bd.setOnes();
    bd *= -1.0;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        Ad(j,0) = point[j].x;
        Ad(j,1) = point[j].y;
        Ad(j,2) = point[j].z;
    }
    Matrix<double, 3, 1> normvecd = Ad.colPivHouseholderQr().solve(bd);
    double nd = normvecd.norm();
    double lambdad = -bd.dot(Ad * normvecd - bd) / (nd * nd * point.size());

    Vector4d pca_resultd;
    pca_resultd(0) = normvecd(0) / nd;
    pca_resultd(1) = normvecd(1) / nd;
    pca_resultd(2) = normvecd(2) / nd;
    pca_resultd(3) = 1.0 / nd;

    bool chk = (lambda > threshold * threshold);
    bool chkd = (lambdad > threshold * threshold);

    if (chk != chkd) {
        cout << "float - double difference occur !!" << endl;
        cout << "---- float ----" << endl;
        cout << "  pca_result: " << pca_result.transpose() << endl;
        cout << "  lambda: " << lambda << endl;
        cout << "---- double ----" << endl;
        cout << "  pca_resultd: " << pca_resultd.transpose() << endl;
        cout << "  lambdad: " << lambdad << endl;
    }

    if (lambda > threshold * threshold)
        return false;


    return true;
}
*/

/*
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const double threshold) {
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }
    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
    T n = normvec.norm();
    T lambda = -b.dot(A * normvec - b) / (n * n * point.size());

    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    if (lambda > threshold * threshold)
        return false;

    return true;
}
*/

/*
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T threshold) {
    Matrix3d H;
    Vector3d r;
    H.setZero();
    r.setZero();

    for (auto &pnt : point) {
        Vector3d p(pnt.x, pnt.y, pnt.z);
        H += p * p.transpose();
        r -= p;
    }
    
    Vector3d center = -r / point.size();
    Vector3d normvec = H.colPivHouseholderQr().solve(r);
    double n = normvec.norm();
    double lambda = (normvec.dot((H - r * r.transpose() / point.size()) * normvec)) / (n * n * point.size());

    if (lambda > threshold * threshold)
        return false;

    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    return true;
}
*/

/*
template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold) {
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    
    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold) {
            return false;
        }
    }
    return true;
    
    
    T lambda = 0;
    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        T dist = pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3);
        lambda += dist * dist;
    }
    lambda /= point.size();

    if (lambda > threshold * threshold) return false;

    return true;  
}
*/

template<typename T>
bool esti_plane_cov(PointPlane &ptpl, const PointCovVector &point, const T threshold) {
    M3D C(M3D::Zero());
    V3D m(V3D::Zero());
    for (auto &p : point) {
        V3D pnt(p.x, p.y, p.z);
        m += pnt;
    }
    m /= point.size();
    ptpl.center = m;
    for (auto &p : point) {
        V3D vec(p.x - m(0), p.y - m(1), p.z - m(2));
        C += vec * vec.transpose();
    }
    C /= point.size();

    JacobiSVD<M3D> svd(C, ComputeFullU | ComputeFullV);
    V3D u1 = svd.matrixU().block<3, 1>(0, 0);
    V3D u2 = svd.matrixU().block<3, 1>(0, 1);
    V3D u3 = svd.matrixU().block<3, 1>(0, 2);
    V3D d = svd.singularValues();

    //==========
    Matrix3f H;
    Vector3f r;
    //Matrix3d H;
    //Vector3d r;
    H.setZero();
    r.setZero();

    for (auto &pnt : point) {
        Vector3f p(pnt.x, pnt.y, pnt.z);
        //Vector3d p(pnt.x, pnt.y, pnt.z);
        H += p * p.transpose();
        r -= p;
    }
    
    Vector3f normvec = H.colPivHouseholderQr().solve(r);
    T n = normvec.norm();
    T lambda = (normvec.dot((H - r * r.transpose() / point.size()) * normvec)) / (n * n * point.size());
    //Vector3d normvec = H.colPivHouseholderQr().solve(r);
    //double n = normvec.norm();
    //double lambda = (normvec.dot((H - r * r.transpose() / point.size()) * normvec)) / (n * n * point.size());
    //============

    
    //if (lambda > threshold * threshold || abs(normvec(2)) < 1.0e-20) return false;
    if (abs(normvec(2)) < 1.0e-20) return false;
    /*
    ptpl.normal(0) = normvec(0) / n;
    ptpl.normal(1) = normvec(1) / n;
    ptpl.normal(2) = normvec(2) / n;
    ptpl.d = 1.0 / n;
    */
    /*
    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (fabs(ptpl.normal(0) * point[j].x + ptpl.normal(1) * point[j].y + ptpl.normal(2) * point[j].z + ptpl.d) > threshold) {
            return false;
        }
    }
    */

    Matrix<T, NUM_MATCH_POINTS, 3> M;
    Matrix<T, NUM_MATCH_POINTS, 1> t;
    M.setZero();
    t.setOnes();
    t *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        M(j,0) = point[j].x;
        M(j,1) = point[j].y;
        M(j,2) = point[j].z;
    }

    Matrix<T, 3, 1> norm_vec = M.colPivHouseholderQr().solve(t);

    T norm = normvec.norm();
    norm_vec /= norm;
    /*
    ptpl.normal(0) = norm_vec(0);
    ptpl.normal(1) = norm_vec(1);
    ptpl.normal(2) = norm_vec(2);
    ptpl.d = 1.0 / norm;
    
    
    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (fabs(norm_vec(0) * point[j].x + norm_vec(1) * point[j].y + norm_vec(2) * point[j].z + 1.0f / norm) > threshold) {
            return false;
        }
    }
    */
    
    if (d(2) > threshold * threshold) return false;
    ptpl.normal = u3;
    ptpl.d = -u3.dot(m);

    /*
    //if (d(2) > threshold * threshold) return false;
    for (int j = 0; j < NUM_MATCH_POINTS; j++) {
        if (fabs(u3(0) * point[j].x + u3(1) * point[j].y + u3(2) * point[j].z + ptpl.d) > threshold) {
            return false;
        }
    }
    */

    M3D A = u1 * u1.transpose() / (d(2) - d(0)) + u2 * u2.transpose() / (d(2) - d(1));
    ptpl.cov_plane.setZero();
    for (auto &p : point) {
        V3D vec(p.x - m(0), p.y - m(1), p.z - m(2));
        M3D B = vec * u3.transpose() + vec.dot(u3) * Eye3d;
        MatrixXd dfdp(6, 3);
        dfdp.setZero();
        dfdp.block<3, 3>(0, 0) = A * B / point.size();
        dfdp(3, 0) = dfdp(4, 1) = dfdp(5, 2) = 1.0 / point.size();

        ptpl.cov_plane += dfdp * p.cov * dfdp.transpose();
    }

    return true;
}

#endif