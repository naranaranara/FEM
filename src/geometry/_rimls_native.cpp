// RIMLS의 값과 역전파 및 헤시안 방향곱을 작은 임시 배열로 계산해요.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>  // ##r## CSR neighbour indices

namespace {
struct Dual {
    double v, d;
    Dual(double value = 0., double tangent = 0.) : v(value), d(tangent) {}
};
// 덧셈의 접선을 전파해요: $$\dot{(a+b)}=\dot a+\dot b$$
inline Dual operator+(Dual a, Dual b) { return {a.v+b.v,a.d+b.d}; }
// 뺄셈의 접선을 전파해요: $$\dot{(a-b)}=\dot a-\dot b$$
inline Dual operator-(Dual a, Dual b) { return {a.v-b.v,a.d-b.d}; }
// 부호 반전의 접선을 전파해요: $$\dot{(-a)}=-\dot a$$
inline Dual operator-(Dual a) { return {-a.v,-a.d}; }
// 곱셈의 접선을 전파해요: $$\dot{(ab)}=\dot a b+a\dot b$$
inline Dual operator*(Dual a, Dual b) { return {a.v*b.v,a.d*b.v+a.v*b.d}; }
inline Dual operator/(Dual a, Dual b) {
    // 나눗셈의 값을 구해요: $$v=a/b$$
    double v=a.v/b.v;
    // 나눗셈의 접선을 전파해요: $$\dot v=(\dot a-v\dot b)/b$$
    return {v,(a.d-v*b.d)/b.v};
}
inline double primal(double x) { return x; }
inline double primal(Dual x) { return x.v; }
inline double tangent(double) { return 0.; }
inline double tangent(Dual x) { return x.d; }
inline double exponential(double x) { return std::exp(x); }
inline Dual exponential(Dual x) {
    // 지수함수의 값을 구해요: $$v=e^x$$
    double v=std::exp(x.v);
    // 지수함수의 접선을 전파해요: $$\dot v=v\dot x$$
    return {v,v*x.d};
}
template<class T> T seeded(double v,double d);
template<> double seeded<double>(double v,double) { return v; }
template<> Dual seeded<Dual>(double v,double d) { return {v,d}; }
template<int B> struct DualPack {
    double v;
    std::array<double,B> d{};
    DualPack(double value=0.) : v(value) {}
    friend DualPack operator+(DualPack a,DualPack b) {
        // 덧셈의 값을 구해요: $$v=a+b$$
        DualPack result(a.v+b.v);
        for(int r=0;r<B;++r) {
            // 각 방향으로 덧셈의 미분을 전파해요: $$\dot v_r=\dot a_r+\dot b_r$$
            result.d[r]=a.d[r]+b.d[r];
        }
        return result;
    }
    friend DualPack operator-(DualPack a,DualPack b) {
        // 뺄셈의 값을 구해요: $$v=a-b$$
        DualPack result(a.v-b.v);
        for(int r=0;r<B;++r) {
            // 각 방향으로 뺄셈의 미분을 전파해요: $$\dot v_r=\dot a_r-\dot b_r$$
            result.d[r]=a.d[r]-b.d[r];
        }
        return result;
    }
    friend DualPack operator-(DualPack a) {
        // 부호 반전의 값을 구해요: $$v=-a$$
        DualPack result(-a.v);
        for(int r=0;r<B;++r) {
            // 각 방향의 미분도 부호를 반전해요: $$\dot v_r=-\dot a_r$$
            result.d[r]=-a.d[r];
        }
        return result;
    }
    friend DualPack operator*(DualPack a,DualPack b) {
        // 곱셈의 값을 구해요: $$v=ab$$
        DualPack result(a.v*b.v);
        for(int r=0;r<B;++r) {
            // 각 방향에 곱의 미분을 전파해요: $$\dot v_r=\dot a_r b+a\dot b_r$$
            result.d[r]=a.d[r]*b.v+a.v*b.d[r];
        }
        return result;
    }
    friend DualPack operator/(DualPack a,DualPack b) {
        // 나눗셈의 값을 구해요: $$v=a/b$$
        DualPack result(a.v/b.v);
        for(int r=0;r<B;++r) {
            // 각 방향에 몫의 미분을 전파해요: $$\dot v_r=(\dot a_r-v\dot b_r)/b$$
            result.d[r]=(a.d[r]-result.v*b.d[r])/b.v;
        }
        return result;
    }
};
template<int B> double primal(const DualPack<B>& x) {return x.v;}
template<int B> DualPack<B> exponential(const DualPack<B>& x) {
    // 모든 방향이 같은 지수함수 값을 공유해요: $$v=e^x$$
    DualPack<B> result(std::exp(x.v));
    for(int r=0;r<B;++r) {
        // 각 방향으로 지수함수의 미분을 전파해요: $$\dot v_r=v\dot x_r$$
        result.d[r]=result.v*x.d[r];
    }
    return result;
}
template<class T> struct DerivativeLanes {
    static constexpr int width=1;
    static T seed(double value,const double* direction,Py_ssize_t index,Py_ssize_t) {
        return seeded<T>(value,direction ? direction[index] : 0.);
    }
    static double get(const T& value,int) {return tangent(value);}
};
template<int B> struct DerivativeLanes<DualPack<B>> {
    static constexpr int width=B;
    static DualPack<B> seed(double value,const double* direction,Py_ssize_t index,Py_ssize_t stride) {
        DualPack<B> result(value);
        if(direction)for(int r=0;r<B;++r)result.d[r]=direction[r*stride+index];
        return result;
    }
    static double get(const DualPack<B>& value,int lane) {return value.d[lane];}
};
template<class T> using Vec=std::array<T,3>;
template<class T> T dot(const Vec<T>& a,const Vec<T>& b) {
    // 세 성분의 내적을 구해요: $$a\cdot b=\sum_{k=1}^3a_kb_k$$
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}
template<class T> struct Pair {
    Vec<T> diff,normal,G,deviation,bar_deviation;
    T t,phi,local,residual,robust,W,bar_W,bar_residual;
};

template<class T> void evaluate(const double* queries,const double* points,const double* normals,
        const double* cotangent,const double* vq,const double* vp,const double* vn,
        double* outq,double* outp,double* outn,double* outg,
        Py_ssize_t Q,Py_ssize_t P,double R,double sr,double sn,
        double* pair_A=nullptr,double* pair_B=nullptr,Py_ssize_t cotangent_batches=1,double* out_values=nullptr) {
    constexpr int lanes=DerivativeLanes<T>::width;
    const Py_ssize_t output_lanes=outg ? lanes : cotangent_batches;
    if(outq)std::fill(outq,outq+output_lanes*3*Q,0.);
    if(outp)std::fill(outp,outp+output_lanes*3*P,0.);
    if(outn)std::fill(outn,outn+output_lanes*3*P,0.);
    // 거리 커널의 반경 제곱을 구해요: $$R_2=R^2$$
    const double R2=R*R;
    // 잔차의 강건 가중치 척도를 구해요: $$a_r=(\sigma_rR)^2$$
    const double ar=(sr*R)*(sr*R);
    // 법선의 강건 가중치 척도를 구해요: $$a_n=\sigma_n^2$$
    const double an=sn*sn;
    std::vector<Pair<T>> cache(P);
    for(Py_ssize_t i=0;i<Q;++i) {
        T D=0.,F=0.;Vec<T> sumG{};
        for(Py_ssize_t j=0;j<P;++j) {
            auto& pair=cache[j];
            for(int k=0;k<3;++k) {
                // 상대좌표와 접선을 구성해요: $$(d_k,\dot d_k)=(q_k-p_k,\dot q_k-\dot p_k)$$
                pair.diff[k]=DerivativeLanes<T>::seed(queries[3*i+k],vq,3*i+k,3*Q)
                           -DerivativeLanes<T>::seed(points[3*j+k],vp,3*j+k,3*P);
                pair.normal[k]=DerivativeLanes<T>::seed(normals[3*j+k],vn,3*j+k,3*P);
            }
            // 거리 커널의 다항식 변수를 구해요: $$t=1-\|d\|^2/R^2$$
            pair.t=1.-dot(pair.diff,pair.diff)/R2;
            if(primal(pair.t)<0.)pair.t=T(0.);
            // 다항식의 제곱을 구해요: $$t_2=t^2$$
            T t2=pair.t*pair.t;
            // 다항식의 세제곱을 구해요: $$t_3=t^3$$
            T t3=t2*pair.t;
            // 거리 가중치를 구해요: $$a=t^4$$
            pair.phi=t2*t2;
            for(int k=0;k<3;++k) {
                // 거리 가중치의 공간 기울기를 구해요: $$G_k=-8t^3d_k/R^2$$
                pair.G[k]=(-8./R2)*t3*pair.diff[k];
                // 공간 기울기의 합을 모아요: $$(\Sigma G)_k\mathrel{+}=G_k$$
                sumG[k]=sumG[k]+pair.G[k];
            }
            // 접평면까지의 부호 거리를 구해요: $$\ell=d\cdot n$$
            pair.local=dot(pair.diff,pair.normal);
            // 초기 거리 가중치의 합을 모아요: $$D\mathrel{+}=a$$
            D=D+pair.phi;
            // 초기 거리의 가중합을 모아요: $$F\mathrel{+}=a\ell$$
            F=F+pair.phi*pair.local;
        }
        // 지지 여부가 반올림에 민감하면 기준 텐서 식에서 판정해요.
        if(out_values && std::abs(primal(D)-1e-14)<=1e-22)throw std::runtime_error("native value boundary fallback");
        if(primal(D)<1e-14)throw std::runtime_error("query outside kernel support");
        // 초기 거리장을 구해요: $$f=F/D$$
        T f=F/D;
        Vec<T> h{};
        for(const auto& pair:cache) {
            for(int k=0;k<3;++k) {
                // 초기 거리장 기울기의 분자를 모아요: $$h_k\mathrel{+}=a n_k+(\ell-f)G_k$$
                h[k]=h[k]+pair.phi*pair.normal[k]+(pair.local-f)*pair.G[k];
            }
        }
        for(int k=0;k<3;++k) {
            // 초기 거리장 기울기를 정규화해요: $$h_k\leftarrow h_k/D$$
            h[k]=h[k]/D;
        }
        T Z=0.,N=0.;
        for(auto& pair:cache) {
            // 접평면에 대한 거리 잔차를 구해요: $$u=f-\ell$$
            pair.residual=f-pair.local;
            for(int k=0;k<3;++k) {
                // 초기 기울기와 법선의 차이를 구해요: $$v_k=h_k-n_k$$
                pair.deviation[k]=h[k]-pair.normal[k];
            }
            // 강건 가중치의 지수 인자를 구해요: $$e=\exp[-u^2/a_r-\|v\|^2/a_n]$$
            pair.robust=exponential(-pair.residual*pair.residual/ar-dot(pair.deviation,pair.deviation)/an);
            // 거리와 강건 가중치를 결합해요: $$w=ae$$
            pair.W=pair.phi*pair.robust;
            // 최종 거리 가중치의 합을 모아요: $$Z\mathrel{+}=w$$
            Z=Z+pair.W;
            // 최종 거리의 가중합을 모아요: $$N\mathrel{+}=w\ell$$
            N=N+pair.W*pair.local;
        }
        // 강건 가중치의 소멸 경계는 기준 텐서 식에서 판정해요.
        if(out_values && std::abs(primal(Z)-1e-14)<=1e-22)throw std::runtime_error("native value boundary fallback");
        if(primal(Z)<=1e-14)throw std::runtime_error("vanishing robust weights");
        // 강건 거리장을 구해요: $$s=N/Z$$
        T value=N/Z;
        if(out_values){out_values[i]=primal(value);if(!pair_A)continue;} // ##r## fused value + first derivatives
        T bar_f=0.;Vec<T> bar_h{};
        if(outg)for(int r=0;r<lanes;++r)outg[r*Q+i]=DerivativeLanes<T>::get(value,r);
        for(auto& pair:cache) {
            // 가중 평균의 가중치 편미분을 구해요: $$\bar w=(\ell-s)/Z$$
            pair.bar_W=(pair.local-value)/Z;
            // 지수 인자의 역전파 계수를 구해요: $$c_e=\bar w w$$
            T coefficient=pair.bar_W*pair.W;
            // 거리 잔차로 역전파해요: $$\bar u=-2c_eu/a_r$$
            pair.bar_residual=(-2./ar)*coefficient*pair.residual;
            // 초기 거리장의 역전파를 모아요: $$\bar f\mathrel{+}=\bar u$$
            bar_f=bar_f+pair.bar_residual;
            for(int k=0;k<3;++k) {
                // 법선 차이로 역전파해요: $$\bar v_k=-2c_ev_k/a_n$$
                pair.bar_deviation[k]=(-2./an)*coefficient*pair.deviation[k];
                // 초기 기울기의 역전파를 모아요: $$\bar h_k\mathrel{+}=\bar v_k$$
                bar_h[k]=bar_h[k]+pair.bar_deviation[k];
            }
        }
        Vec<T> c{};
        for(int k=0;k<3;++k) {
            // 기울기 정규화의 역전파 계수를 구해요: $$c_k=\bar h_k/D$$
            c[k]=bar_h[k]/D;
        }
        // 기울기를 경유한 거리장 기여를 더해요: $$\bar f\leftarrow\bar f-c\cdot\sum_jG_j$$
        bar_f=bar_f-dot(c,sumG);
        // 두 정규화의 분모로 역전파해요: $$\bar D=-\bar h\cdot h/D-\bar f f/D$$
        T bar_D=-dot(bar_h,h)/D-bar_f*f/D;
        for(Py_ssize_t j=0;j<P;++j) {
            const auto& pair=cache[j];
            // 초기 거리 가중치로 역전파해요: $$\bar a=\bar w e+c\cdot n+\ell\bar f/D+\bar D$$
            T bar_phi=pair.bar_W*pair.robust+dot(c,pair.normal)+pair.local*(bar_f/D)+bar_D;
            // 접평면 거리로 역전파해요: $$\bar\ell=w/Z-\bar u+c\cdot G+a\bar f/D$$
            T bar_local=pair.W/Z-pair.bar_residual+dot(c,pair.G)+pair.phi*(bar_f/D);
            Vec<T> bar_G{};
            for(int k=0;k<3;++k) {
                // 가중치 기울기로 역전파해요: $$\bar G_k=(\ell-f)c_k$$
                bar_G[k]=(pair.local-f)*c[k];
            }
            // 다항식의 제곱을 구해요: $$t_2=t^2$$
            T t2=pair.t*pair.t;
            // 다항식의 세제곱을 구해요: $$t_3=t^3$$
            T t3=t2*pair.t;
            // 다항식 변수로 역전파해요: $$\bar t=4t^3\bar a-24t^2(\bar G\cdot d)/R^2$$
            T bar_t=4.*t3*bar_phi-(24./R2)*t2*dot(bar_G,pair.diff);
            for(int k=0;k<3;++k) {
                // 상대좌표의 편미분을 구해요: $$A_k=-8t^3\bar G_k/R^2-2\bar t d_k/R^2+\bar\ell n_k$$
                T A=(-8./R2)*t3*bar_G[k]-(2./R2)*bar_t*pair.diff[k]+bar_local*pair.normal[k];
                // 법선의 편미분을 구해요: $$B_k=-\bar v_k+a c_k+\bar\ell d_k$$
                T B=-pair.bar_deviation[k]+pair.phi*c[k]+bar_local*pair.diff[k];
                if(pair_A) {
                    pair_A[3*(i*P+j)+k]=primal(A);
                    pair_B[3*(i*P+j)+k]=primal(B);
                    continue;
                }
                for(Py_ssize_t r=0;r<output_lanes;++r) {
                    const double weight=outg ? cotangent[i] : cotangent[r*Q+i];
                    // 좌표 편미분에 출력의 역전파 가중치를 곱해요: $$g_A=\bar s_i(A_k\text{ 또는 }\dot A_{rk})$$
                    double ga=weight*(outg ? DerivativeLanes<T>::get(A,r) : primal(A));
                    // 법선 편미분에 출력의 역전파 가중치를 곱해요: $$g_B=\bar s_i(B_k\text{ 또는 }\dot B_{rk})$$
                    double gb=weight*(outg ? DerivativeLanes<T>::get(B,r) : primal(B));
                    // 질의 좌표의 역전파를 모아요: $$\bar q_{rik}\mathrel{+}=g_A$$
                    outq[r*3*Q+3*i+k]+=ga;
                    // 표면 좌표의 역전파를 모아요: $$\bar p_{rjk}\mathrel{-}=g_A$$
                    outp[r*3*P+3*j+k]-=ga;
                    // 표면 법선의 역전파를 모아요: $$\bar n_{rjk}\mathrel{+}=g_B$$
                    outn[r*3*P+3*j+k]+=gb;
                }
            }
        }
    }
}

struct Buffer {
    Py_buffer value{}; bool acquired=false;
    ~Buffer(){if(acquired)PyBuffer_Release(&value);}
    bool get(PyObject* object,bool writable) {
        int flags=PyBUF_FORMAT|PyBUF_C_CONTIGUOUS|(writable ? PyBUF_WRITABLE : 0);
        if(PyObject_GetBuffer(object,&value,flags)<0)return false;
        acquired=true;
        if(value.itemsize!=8 || !value.format || std::strcmp(value.format,"d")!=0) {
            PyErr_SetString(PyExc_TypeError,"native RIMLS requires contiguous native float64 buffers");return false;
        }
        return true;
    }
    double* data(){return static_cast<double*>(value.buf);}
    Py_ssize_t size(){return value.len/8;}
};

PyObject* call(PyObject* args,bool hessian) {
    const int objects=hessian ? 11 : 7;
    if(PyTuple_Size(args)!=objects+3) {
        PyErr_SetString(PyExc_TypeError,"invalid native RIMLS argument count");return nullptr;
    }
    Buffer b[11];
    for(int i=0;i<objects;++i)
        if(!b[i].get(PyTuple_GET_ITEM(args,i),i>=(hessian ? 7 : 4)))return nullptr;
    double R=PyFloat_AsDouble(PyTuple_GET_ITEM(args,objects));
    double sr=PyFloat_AsDouble(PyTuple_GET_ITEM(args,objects+1));
    double sn=PyFloat_AsDouble(PyTuple_GET_ITEM(args,objects+2));
    if(PyErr_Occurred())return nullptr;
    if(!(R>0 && sr>0 && sn>0)) {PyErr_SetString(PyExc_ValueError,"positive R and robust scales required");return nullptr;}
    const Py_ssize_t Q=b[0].size()/3,P=b[1].size()/3;
    if(b[0].size()!=3*Q || b[1].size()!=3*P || b[2].size()!=3*P || b[3].size()!=Q) {
        PyErr_SetString(PyExc_ValueError,"inconsistent native RIMLS input sizes");return nullptr;
    }
    const int out=hessian ? 7 : 4;
    if(b[out].size()!=3*Q || b[out+1].size()!=3*P || b[out+2].size()!=3*P ||
       (hessian && (b[4].size()!=3*Q || b[5].size()!=3*P || b[6].size()!=3*P || b[10].size()!=Q))) {
        PyErr_SetString(PyExc_ValueError,"inconsistent native RIMLS output/direction sizes");return nullptr;
    }
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try {
        if(hessian)evaluate<Dual>(b[0].data(),b[1].data(),b[2].data(),b[3].data(),
                    b[4].data(),b[5].data(),b[6].data(),b[7].data(),b[8].data(),b[9].data(),b[10].data(),Q,P,R,sr,sn);
        else evaluate<double>(b[0].data(),b[1].data(),b[2].data(),b[3].data(),nullptr,nullptr,nullptr,
                    b[4].data(),b[5].data(),b[6].data(),nullptr,Q,P,R,sr,sn);
    } catch(const std::exception& exc) {error=exc.what();}
    Py_END_ALLOW_THREADS
    if(!error.empty()){PyErr_SetString(PyExc_ValueError,error.c_str());return nullptr;}
    Py_RETURN_NONE;
}
template<class T> bool evaluate_batch(Buffer* b,Py_ssize_t Q,Py_ssize_t P,double R,double sr,double sn) {
    // 여러 방향의 임시 배열이 정해진 작업공간을 넘으면 스칼라 경로를 사용해요.
    constexpr std::size_t workspace_limit=64*1024*1024;
    if(DerivativeLanes<T>::width>1 && static_cast<std::size_t>(P)>workspace_limit/sizeof(Pair<T>))return false;
    evaluate<T>(b[0].data(),b[1].data(),b[2].data(),b[3].data(),
                b[4].data(),b[5].data(),b[6].data(),b[7].data(),b[8].data(),b[9].data(),b[10].data(),Q,P,R,sr,sn);
    return true;
}
PyObject* hvp_batch(PyObject*,PyObject* args) {
    if(PyTuple_Size(args)!=14) {
        PyErr_SetString(PyExc_TypeError,"invalid native RIMLS batch argument count");return nullptr;
    }
    Buffer b[11];
    for(int i=0;i<11;++i)
        if(!b[i].get(PyTuple_GET_ITEM(args,i),i>=7))return nullptr;
    double R=PyFloat_AsDouble(PyTuple_GET_ITEM(args,11));
    double sr=PyFloat_AsDouble(PyTuple_GET_ITEM(args,12));
    double sn=PyFloat_AsDouble(PyTuple_GET_ITEM(args,13));
    if(PyErr_Occurred())return nullptr;
    if(!(R>0 && sr>0 && sn>0)) {
        PyErr_SetString(PyExc_ValueError,"positive R and robust scales required");return nullptr;
    }
    const Py_ssize_t Q=b[0].size()/3,P=b[1].size()/3;
    if(b[4].value.ndim!=3) {
        PyErr_SetString(PyExc_ValueError,"batched RIMLS directions require shape (batch,queries,3)");return nullptr;
    }
    const Py_ssize_t B=b[4].value.shape[0];
    if(B<0 || b[0].size()!=3*Q || b[1].size()!=3*P || b[2].size()!=3*P || b[3].size()!=Q ||
       (Q && B>PY_SSIZE_T_MAX/(3*Q)) || (P && B>PY_SSIZE_T_MAX/(3*P)) ||
       b[4].value.shape[1]!=Q || b[4].value.shape[2]!=3 ||
       b[4].size()!=B*3*Q || b[5].size()!=B*3*P || b[6].size()!=B*3*P ||
       b[7].size()!=B*3*Q || b[8].size()!=B*3*P || b[9].size()!=B*3*P || b[10].size()!=B*Q) {
        PyErr_SetString(PyExc_ValueError,"inconsistent native RIMLS batch sizes");return nullptr;
    }
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try {
        bool complete=false;
        switch(B) {
            case 1:complete=evaluate_batch<Dual>(b,Q,P,R,sr,sn);break;
            case 2:complete=evaluate_batch<DualPack<2>>(b,Q,P,R,sr,sn);break;
            case 4:complete=evaluate_batch<DualPack<4>>(b,Q,P,R,sr,sn);break;
            case 8:complete=evaluate_batch<DualPack<8>>(b,Q,P,R,sr,sn);break;
            case 16:complete=evaluate_batch<DualPack<16>>(b,Q,P,R,sr,sn);break;
        }
        if(!complete)
            for(Py_ssize_t r=0;r<B;++r)
                evaluate<Dual>(b[0].data(),b[1].data(),b[2].data(),b[3].data(),
                    b[4].data()+r*3*Q,b[5].data()+r*3*P,b[6].data()+r*3*P,
                    b[7].data()+r*3*Q,b[8].data()+r*3*P,b[9].data()+r*3*P,b[10].data()+r*Q,Q,P,R,sr,sn);
    } catch(const std::exception& exc) {error=exc.what();}
    Py_END_ALLOW_THREADS
    if(!error.empty()){PyErr_SetString(PyExc_ValueError,error.c_str());return nullptr;}
    Py_RETURN_NONE;
}
PyObject* vjp_batch(PyObject*,PyObject* args) {
    if(PyTuple_Size(args)!=10) {
        PyErr_SetString(PyExc_TypeError,"invalid native RIMLS batch argument count");return nullptr;
    }
    Buffer b[7];
    for(int i=0;i<7;++i)
        if(!b[i].get(PyTuple_GET_ITEM(args,i),i>=4))return nullptr;
    double R=PyFloat_AsDouble(PyTuple_GET_ITEM(args,7));
    double sr=PyFloat_AsDouble(PyTuple_GET_ITEM(args,8));
    double sn=PyFloat_AsDouble(PyTuple_GET_ITEM(args,9));
    if(PyErr_Occurred())return nullptr;
    if(!(R>0 && sr>0 && sn>0)) {
        PyErr_SetString(PyExc_ValueError,"positive R and robust scales required");return nullptr;
    }
    const Py_ssize_t Q=b[0].size()/3,P=b[1].size()/3;
    if(b[3].value.ndim!=2) {
        PyErr_SetString(PyExc_ValueError,"batched RIMLS cotangents require shape (batch,queries)");return nullptr;
    }
    const Py_ssize_t B=b[3].value.shape[0];
    if(B<0 || b[0].size()!=3*Q || b[1].size()!=3*P || b[2].size()!=3*P ||
       (Q && B>PY_SSIZE_T_MAX/(3*Q)) || (P && B>PY_SSIZE_T_MAX/(3*P)) ||
       b[3].value.shape[1]!=Q || b[3].size()!=B*Q ||
       b[4].size()!=B*3*Q || b[5].size()!=B*3*P || b[6].size()!=B*3*P) {
        PyErr_SetString(PyExc_ValueError,"inconsistent native RIMLS batch sizes");return nullptr;
    }
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try {
        if(B)evaluate<double>(b[0].data(),b[1].data(),b[2].data(),b[3].data(),nullptr,nullptr,nullptr,
                              b[4].data(),b[5].data(),b[6].data(),nullptr,Q,P,R,sr,sn,nullptr,nullptr,B);
    } catch(const std::exception& exc) {error=exc.what();}
    Py_END_ALLOW_THREADS
    if(!error.empty()){PyErr_SetString(PyExc_ValueError,error.c_str());return nullptr;}
    Py_RETURN_NONE;
}
PyObject* pair_jacobian(PyObject*,PyObject* args) {
    if(PyTuple_Size(args)!=8) {
        PyErr_SetString(PyExc_TypeError,"invalid native RIMLS pair argument count");return nullptr;
    }
    Buffer b[5];
    for(int i=0;i<5;++i)
        if(!b[i].get(PyTuple_GET_ITEM(args,i),i>=3))return nullptr;
    double R=PyFloat_AsDouble(PyTuple_GET_ITEM(args,5));
    double sr=PyFloat_AsDouble(PyTuple_GET_ITEM(args,6));
    double sn=PyFloat_AsDouble(PyTuple_GET_ITEM(args,7));
    if(PyErr_Occurred())return nullptr;
    if(!(R>0 && sr>0 && sn>0)) {
        PyErr_SetString(PyExc_ValueError,"positive R and robust scales required");return nullptr;
    }
    const Py_ssize_t Q=b[0].size()/3,P=b[1].size()/3;
    if(b[0].size()!=3*Q || b[1].size()!=3*P || b[2].size()!=3*P ||
       (Q && P>PY_SSIZE_T_MAX/(3*Q)) || b[3].size()!=3*Q*P || b[4].size()!=3*Q*P) {
        PyErr_SetString(PyExc_ValueError,"inconsistent native RIMLS pair sizes");return nullptr;
    }
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try {
        evaluate<double>(b[0].data(),b[1].data(),b[2].data(),nullptr,nullptr,nullptr,nullptr,
                         nullptr,nullptr,nullptr,nullptr,Q,P,R,sr,sn,b[3].data(),b[4].data());
    } catch(const std::exception& exc) {error=exc.what();}
    Py_END_ALLOW_THREADS
    if(!error.empty()){PyErr_SetString(PyExc_ValueError,error.c_str());return nullptr;}
    Py_RETURN_NONE;
}
PyObject* values(PyObject*,PyObject* args) {
    if(PyTuple_Size(args)!=7) {
        PyErr_SetString(PyExc_TypeError,"invalid native RIMLS value argument count");return nullptr;
    }
    Buffer b[4];
    for(int i=0;i<4;++i)if(!b[i].get(PyTuple_GET_ITEM(args,i),i==3))return nullptr;
    double R=PyFloat_AsDouble(PyTuple_GET_ITEM(args,4));
    double sr=PyFloat_AsDouble(PyTuple_GET_ITEM(args,5));
    double sn=PyFloat_AsDouble(PyTuple_GET_ITEM(args,6));
    if(PyErr_Occurred())return nullptr;
    if(!(R>0 && sr>0 && sn>0)) {PyErr_SetString(PyExc_ValueError,"positive R and robust scales required");return nullptr;}
    const Py_ssize_t Q=b[0].size()/3,P=b[1].size()/3;
    if(b[0].value.ndim!=2 || b[0].value.shape[1]!=3 ||
       b[1].value.ndim!=2 || b[1].value.shape[1]!=3 ||
       b[2].value.ndim!=2 || b[2].value.shape[1]!=3 ||
       b[3].value.ndim!=1 || b[0].size()!=3*Q || b[1].size()!=3*P || b[2].size()!=3*P || b[3].size()!=Q) {
        PyErr_SetString(PyExc_ValueError,"inconsistent native RIMLS value shapes");return nullptr;
    }
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try {
        evaluate<double>(b[0].data(),b[1].data(),b[2].data(),nullptr,nullptr,nullptr,nullptr,
                         nullptr,nullptr,nullptr,nullptr,Q,P,R,sr,sn,nullptr,nullptr,1,b[3].data());
    } catch(const std::exception& exc) {error=exc.what();}
    Py_END_ALLOW_THREADS
    if(!error.empty()){PyErr_SetString(PyExc_ValueError,error.c_str());return nullptr;}
    Py_RETURN_NONE;
}

// ##r## Compact CSR neighbourhoods: evaluate each actual query/sample pair once.
struct IndexBuffer {
    Py_buffer value{}; bool acquired=false;
    ~IndexBuffer(){if(acquired)PyBuffer_Release(&value);}
    bool get(PyObject* object) {
        if(PyObject_GetBuffer(object,&value,PyBUF_FORMAT|PyBUF_C_CONTIGUOUS)<0)return false;
        acquired=true;
        if(value.ndim!=1 || value.itemsize!=8 || !value.format ||
           (std::strcmp(value.format,"q")!=0 && std::strcmp(value.format,"l")!=0)) {
            PyErr_SetString(PyExc_TypeError,"native neighbours require native int64 vectors");return false;
        }
        return true;
    }
    const std::int64_t* data(){return static_cast<const std::int64_t*>(value.buf);}
    Py_ssize_t size(){return value.len/8;}
};

PyObject* local_evaluate(PyObject*,PyObject* args) {
    if(PyTuple_Size(args)!=11) {
        PyErr_SetString(PyExc_TypeError,"invalid local RIMLS argument count");return nullptr;
    }
    Buffer b[6];IndexBuffer offsets,ids;
    const int positions[]={0,1,2,5,6,7};
    for(int i=0;i<6;++i)if(!b[i].get(PyTuple_GET_ITEM(args,positions[i]),i>=3))return nullptr;
    if(!offsets.get(PyTuple_GET_ITEM(args,3)) || !ids.get(PyTuple_GET_ITEM(args,4)))return nullptr;
    double R=PyFloat_AsDouble(PyTuple_GET_ITEM(args,8));
    double sr=PyFloat_AsDouble(PyTuple_GET_ITEM(args,9));
    double sn=PyFloat_AsDouble(PyTuple_GET_ITEM(args,10));
    if(PyErr_Occurred())return nullptr;
    if(!(R>0 && sr>0 && sn>0 && std::isfinite(R) && std::isfinite(sr) && std::isfinite(sn))) {
        PyErr_SetString(PyExc_ValueError,"positive finite scales required");return nullptr;
    }
    const Py_ssize_t Q=b[0].size()/3,P=b[1].size()/3,N=ids.size();
    bool derivatives=b[4].size()!=0 || b[5].size()!=0;
    if(b[0].value.ndim!=2 || b[0].value.shape[1]!=3 ||
       b[1].value.ndim!=2 || b[1].value.shape[1]!=3 ||
       b[2].value.ndim!=2 || b[2].value.shape[1]!=3 || b[2].size()!=3*P ||
       b[3].size()!=Q || offsets.size()!=Q+1 || N>PY_SSIZE_T_MAX/3 ||
       (derivatives && (b[4].size()!=3*N || b[5].size()!=3*N))) {
        PyErr_SetString(PyExc_ValueError,"inconsistent local RIMLS buffers");return nullptr;
    }
    const auto* ptr=offsets.data();const auto* ix=ids.data();
    if(ptr[0]!=0 || ptr[Q]!=N) {
        PyErr_SetString(PyExc_ValueError,"invalid neighbour offsets");return nullptr;
    }
    for(Py_ssize_t i=0;i<Q;++i) {
        if(ptr[i]<0 || ptr[i+1]<ptr[i] || ptr[i+1]>N) {
            PyErr_SetString(PyExc_ValueError,"invalid neighbour offsets");return nullptr;
        }
        for(auto j=ptr[i];j<ptr[i+1];++j)if(ix[j]<0 || ix[j]>=P || (j>ptr[i] && ix[j]<=ix[j-1])) {
            PyErr_SetString(PyExc_ValueError,"invalid or unsorted neighbour index");return nullptr;
        }
    }
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::vector<double> points,normals;
        for(Py_ssize_t i=0;i<Q;++i) {
            const Py_ssize_t start=ptr[i],count=ptr[i+1]-start;
            points.resize(3*count);normals.resize(3*count);
            for(Py_ssize_t j=0;j<count;++j)for(int k=0;k<3;++k) {
                points[3*j+k]=b[1].data()[3*ix[start+j]+k];
                normals[3*j+k]=b[2].data()[3*ix[start+j]+k];
            }
            evaluate<double>(b[0].data()+3*i,points.data(),normals.data(),nullptr,nullptr,nullptr,nullptr,
                nullptr,nullptr,nullptr,nullptr,1,count,R,sr,sn,
                derivatives ? b[4].data()+3*start : nullptr,
                derivatives ? b[5].data()+3*start : nullptr,1,b[3].data()+i);
        }
    } catch(const std::exception& exc){error=exc.what();}
    Py_END_ALLOW_THREADS
    if(!error.empty()){PyErr_SetString(PyExc_ValueError,error.c_str());return nullptr;}
    Py_RETURN_NONE;
}

PyObject* vjp(PyObject*,PyObject* args){return call(args,false);}
PyObject* hvp(PyObject*,PyObject* args){return call(args,true);}
PyMethodDef methods[]={{"local_evaluate",local_evaluate,METH_VARARGS,"##r## Neighbour-only values and first derivatives."},{"values",values,METH_VARARGS,"역전파가 필요 없는 RIMLS 간격을 계산해요."},{"vjp",vjp,METH_VARARGS,"좌표와 법선의 역전파를 계산해요."},
                      {"hvp",hvp,METH_VARARGS,"헤시안 방향곱을 계산해요."},
                      {"hvp_batch",hvp_batch,METH_VARARGS,"여러 방향의 헤시안 곱에서 공통 계산을 공유해요."},
                      {"vjp_batch",vjp_batch,METH_VARARGS,"여러 역전파 가중치가 쌍별 편미분 계산을 공유해요."},
                      {"pair_jacobian",pair_jacobian,METH_VARARGS,"좌표와 법선의 쌍별 편미분을 보관해요."},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_rimls_native","RIMLS의 정확한 네이티브 미분을 제공해요.",-1,methods};
}
PyMODINIT_FUNC PyInit__rimls_native(){return PyModule_Create(&module);}
