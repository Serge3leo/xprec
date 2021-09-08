#include "Python.h"
#include "math.h"
#include "stdio.h"

#include "dd_arith.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"


/**
 * Allows parameter to be marked unused
 */
#define MARK_UNUSED(x)  do { (void)(x); } while(false)

/**
 * Create ufunc loop routine for a unary operation
 */
#define ULOOP_UNARY(func_name, inner_func, type_out, type_in)           \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp *steps, void *data)            \
    {                                                                   \
        assert (sizeof(ddouble) == 2 * sizeof(double));                 \
        const npy_intp n = dimensions[0];                               \
        const npy_intp is1 = steps[0], os1 = steps[1];                  \
        char *_in1 = args[0], *_out1 = args[1];                         \
                                                                        \
        for (npy_intp i = 0; i < n; i++) {                              \
            const type_in *in = (const type_in *)_in1;                  \
            type_out *out = (type_out *)_out1;                          \
            *out = inner_func(*in);                                     \
                                                                        \
            _in1 += is1;                                                \
            _out1 += os1;                                               \
        }                                                               \
        MARK_UNUSED(data);                                              \
    }

/**
 * Create ufunc loop routine for a binary operation
 */
#define ULOOP_BINARY(func_name, inner_func, type_r, type_a, type_b)     \
    static void func_name(char **args, const npy_intp *dimensions,      \
                          const npy_intp* steps, void *data)            \
    {                                                                   \
        assert (sizeof(ddouble) == 2 * sizeof(double));                 \
        const npy_intp n = dimensions[0];                               \
        const npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];  \
        char *_in1 = args[0], *_in2 = args[1], *_out1 = args[2];        \
                                                                        \
        for (npy_intp i = 0; i < n; i++) {                              \
            const type_a *lhs = (const type_a *)_in1;                   \
            const type_b *rhs = (const type_b *)_in2;                   \
            type_r *out = (type_r *)_out1;                              \
            *out = inner_func(*lhs, *rhs);                              \
                                                                        \
            _in1 += is1;                                                \
            _in2 += is2;                                                \
            _out1 += os1;                                               \
        }                                                               \
        MARK_UNUSED(data);                                              \
    }

ULOOP_BINARY(u_addqd, addqd, ddouble, ddouble, double)
ULOOP_BINARY(u_subqd, subqd, ddouble, ddouble, double)
ULOOP_BINARY(u_mulqd, mulqd, ddouble, ddouble, double)
ULOOP_BINARY(u_divqd, divqd, ddouble, ddouble, double)
ULOOP_BINARY(u_adddq, adddq, ddouble, double, ddouble)
ULOOP_BINARY(u_subdq, subdq, ddouble, double, ddouble)
ULOOP_BINARY(u_muldq, muldq, ddouble, double, ddouble)
ULOOP_BINARY(u_divdq, divdq, ddouble, double, ddouble)

ULOOP_BINARY(u_addqq, addqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_subqq, subqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_mulqq, mulqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_divqq, divqq, ddouble, ddouble, ddouble)

ULOOP_UNARY(u_negq, negq, ddouble, ddouble)
ULOOP_UNARY(u_posq, posq, ddouble, ddouble)
ULOOP_UNARY(u_absq, absq, ddouble, ddouble)
ULOOP_UNARY(u_reciprocalq, reciprocalq, ddouble, ddouble)
ULOOP_UNARY(u_sqrq, sqrq, ddouble, ddouble)
ULOOP_UNARY(u_roundq, roundq, ddouble, ddouble)
ULOOP_UNARY(u_floorq, floorq, ddouble, ddouble)
ULOOP_UNARY(u_ceilq, ceilq, ddouble, ddouble)

ULOOP_UNARY(u_signbitq, signbitq, bool, ddouble)
ULOOP_BINARY(u_copysignqq, copysignqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_copysignqd, copysignqd, ddouble, ddouble, double)
ULOOP_BINARY(u_copysigndq, copysigndq, ddouble, double, ddouble)
ULOOP_UNARY(u_signq, signq, ddouble, ddouble)

ULOOP_UNARY(u_isfiniteq, isfiniteq, bool, ddouble)
ULOOP_UNARY(u_isinfq, isinfq, bool, ddouble)
ULOOP_UNARY(u_isnanq, isnanq, bool, ddouble)

ULOOP_BINARY(u_equalqq, equalqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_notequalqq, notequalqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_greaterqq, greaterqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_lessqq, lessqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_greaterequalqq, greaterqq, bool, ddouble, ddouble)
ULOOP_BINARY(u_lessequalqq, lessqq, bool, ddouble, ddouble)

ULOOP_BINARY(u_equalqd, equalqd, bool, ddouble, double)
ULOOP_BINARY(u_notequalqd, notequalqd, bool, ddouble, double)
ULOOP_BINARY(u_greaterqd, greaterqd, bool, ddouble, double)
ULOOP_BINARY(u_lessqd, lessqd, bool, ddouble, double)
ULOOP_BINARY(u_greaterequalqd, greaterequalqd, bool, ddouble, double)
ULOOP_BINARY(u_lessequalqd, lessequalqd, bool, ddouble, double)

ULOOP_BINARY(u_equaldq, equaldq, bool, double, ddouble)
ULOOP_BINARY(u_notequaldq, notequaldq, bool, double, ddouble)
ULOOP_BINARY(u_greaterdq, greaterdq, bool, double, ddouble)
ULOOP_BINARY(u_lessdq, lessdq, bool, double, ddouble)
ULOOP_BINARY(u_greaterequaldq, greaterequaldq, bool, double, ddouble)
ULOOP_BINARY(u_lessequaldq, lessequaldq, bool, double, ddouble)

ULOOP_BINARY(u_fminqq, fminqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_fmaxqq, fmaxqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_fminqd, fminqd, ddouble, ddouble, double)
ULOOP_BINARY(u_fmaxqd, fmaxqd, ddouble, ddouble, double)
ULOOP_BINARY(u_fmindq, fmindq, ddouble, double, ddouble)
ULOOP_BINARY(u_fmaxdq, fmaxdq, ddouble, double, ddouble)

ULOOP_UNARY(u_iszeroq, iszeroq, bool, ddouble)
ULOOP_UNARY(u_isoneq, isoneq, bool, ddouble)
ULOOP_UNARY(u_ispositiveq, ispositiveq, bool, ddouble)
ULOOP_UNARY(u_isnegativeq, isnegativeq, bool, ddouble)

ULOOP_UNARY(u_sqrtq, sqrtq, ddouble, ddouble)

/* Inverse Factorials from 1/3!, 1/4!, asf. */
static int _n_inv_fact = 15;
static const ddouble _inv_fact[] = {
    {1.66666666666666657e-01, 9.25185853854297066e-18},
    {4.16666666666666644e-02, 2.31296463463574266e-18},
    {8.33333333333333322e-03, 1.15648231731787138e-19},
    {1.38888888888888894e-03, -5.30054395437357706e-20},
    {1.98412698412698413e-04, 1.72095582934207053e-22},
    {2.48015873015873016e-05, 2.15119478667758816e-23},
    {2.75573192239858925e-06, -1.85839327404647208e-22},
    {2.75573192239858883e-07, 2.37677146222502973e-23},
    {2.50521083854417202e-08, -1.44881407093591197e-24},
    {2.08767569878681002e-09, -1.20734505911325997e-25},
    {1.60590438368216133e-10, 1.25852945887520981e-26},
    {1.14707455977297245e-11, 2.06555127528307454e-28},
    {7.64716373181981641e-13, 7.03872877733453001e-30},
    {4.77947733238738525e-14, 4.39920548583408126e-31},
    {2.81145725434552060e-15, 1.65088427308614326e-31}
    };

/**
 * For the exponential of `a`, return compute tuple `x, m` such that:
 *
 *      exp(a) = ldexp(1 + x, m),
 *
 * where `m` is chosen such that `abs(x) < 1`.  The value `x` is returned,
 * whereas the value `m` is given as an out parameter.
 */
static ddouble _exp_reduced(ddouble a, int *m)
{
    /* Strategy:  We first reduce the size of x by noting that
     *
     *     exp(k * r + m * log(2)) = 2^m * exp(r)^k
     *
     * where m and k are integers.  By choosing m appropriately
     * we can make |k * r| <= log(2) / 2 = 0.347.
     */
    const double k = 512.0;
    const double inv_k = 1.0 / k;
    double mm = floor(a.hi / Q_LOG2.hi + 0.5);
    ddouble r = mul_pwr2(subqq(a, mulqd(Q_LOG2, mm)), inv_k);
    *m = (int)mm;

    /* Now, evaluate exp(r) using the familiar Taylor series.  Reducing the
     * argument substantially speeds up the convergence.  First, we compute
     * terms of order 1 and 2 and add it to the sum
     */
    ddouble sum, term, rpower;
    rpower = sqrq(r);
    sum = addqq(r, mul_pwr2(rpower, 0.5));

    /* Next, compute terms of order 3 and up */
    rpower = mulqq(rpower, r);
    term = mulqq(rpower, _inv_fact[0]);
    int i = 0;
    do {
        sum = addqq(sum, term);
        rpower = mulqq(rpower, r);
        ++i;
        term = mulqq(rpower, _inv_fact[i]);
    } while (fabs(term.hi) > inv_k * Q_EPS.hi && i < 5);
    sum = addqq(sum, term);

    /* We now have that approximately exp(r) == 1 + sum.  Raise that to
     * the m'th (512) power by squaring the binomial nine times
     */
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    sum = addqq(mul_pwr2(sum, 2.0), sqrq(sum));
    return sum;
}

static ddouble expq(ddouble a)
{
    if (a.hi <= -709.0)
        return Q_ZERO;
    if (a.hi >= 709.0)
        return infq();
    if (iszeroq(a))
        return Q_ONE;
    if (isoneq(a))
        return Q_E;

    int m;
    ddouble sum = _exp_reduced(a, &m);

    /** Add back the one and multiply by 2 to the m */
    sum = addqd(sum, 1.0);
    return ldexpq(sum, (int)m);
}
ULOOP_UNARY(u_expq, expq, ddouble, ddouble)

static ddouble expm1q(ddouble a)
{
    if (a.hi <= -709.0)
        return (ddouble){-1.0, 0.0};
    if (a.hi >= 709.0)
        return infq();
    if (iszeroq(a))
        return Q_ZERO;

    int m;
    ddouble sum = _exp_reduced(a, &m);

    /* Truncation case: simply return sum */
    if (m == 0)
        return sum;

    /* Non-truncation case: compute full exp, then remove the one */
    sum = addqd(sum, 1.0);
    sum = ldexpq(sum, (int)m);
    return subqd(sum, 1.0);
}
ULOOP_UNARY(u_expm1q, expm1q, ddouble, ddouble)

static ddouble logq(ddouble a)
{
    /* Strategy.  The Taylor series for log converges much more
     * slowly than that of exp, due to the lack of the factorial
     * term in the denominator.  Hence this routine instead tries
     * to determine the root of the function
     *
     *     f(x) = exp(x) - a
     *
     * using Newton iteration.  The iteration is given by
     *
     *     x' = x - f(x)/f'(x)
     *        = x - (1 - a * exp(-x))
     *        = x + a * exp(-x) - 1.
     *
     * Only one iteration is needed, since Newton's iteration
     * approximately doubles the number of digits per iteration.
     */
    if (isoneq(a))
        return Q_ZERO;
    if (a.hi <= 0.0)
        return nanq();

    ddouble x = {log(a.hi), 0.0}; /* Initial approximation */
    x = subqd(addqq(x, mulqq(a, expq(negq(x)))), 1.0);
    return x;
}
ULOOP_UNARY(u_logq, logq, ddouble, ddouble)

static const ddouble _pi_16 =
    {1.963495408493620697e-01, 7.654042494670957545e-18};

/* Table of sin(k * pi/16) and cos(k * pi/16). */
static const ddouble _sin_table[] = {
    {1.950903220161282758e-01, -7.991079068461731263e-18},
    {3.826834323650897818e-01, -1.005077269646158761e-17},
    {5.555702330196021776e-01, 4.709410940561676821e-17},
    {7.071067811865475727e-01, -4.833646656726456726e-17}
    };

static const ddouble _cos_table[] = {
    {9.807852804032304306e-01, 1.854693999782500573e-17},
    {9.238795325112867385e-01, 1.764504708433667706e-17},
    {8.314696123025452357e-01, 1.407385698472802389e-18},
    {7.071067811865475727e-01, -4.833646656726456726e-17}
    };

static ddouble sin_taylor(ddouble a)
{
    const double thresh = 0.5 * fabs(a.hi) * Q_EPS.hi;
    ddouble r, s, t, x;

    if (iszeroq(a))
        return Q_ZERO;

    int i = 0;
    x = negq(sqrq(a));
    s = a;
    r = a;
    do {
        r = mulqq(r, x);
        t = mulqq(r, _inv_fact[i]);
        s = addqq(s, t);
        i += 2;
    } while (i < _n_inv_fact && fabs(t.hi) > thresh);

    return s;
}

static ddouble cos_taylor(ddouble a)
{
    const double thresh = 0.5 * Q_EPS.hi;
    ddouble r, s, t, x;

    if (iszeroq(a))
        return Q_ONE;

    x = negq(sqrq(a));
    r = x;
    s = adddq(1.0, mul_pwr2(r, 0.5));
    int i = 1;
    do {
        r = mulqq(r, x);
        t = mulqq(r, _inv_fact[i]);
        s = addqq(s, t);
        i += 2;
    } while (i < _n_inv_fact && fabs(t.hi) > thresh);

    return s;
}

static void sincos_taylor(ddouble a, ddouble *sin_a, ddouble *cos_a)
{
    if (iszeroq(a)) {
        *sin_a = Q_ZERO;
        *cos_a = Q_ONE;
    } else {
        *sin_a = sin_taylor(a);
        *cos_a = sqrtq(subdq(1.0, sqrq(*sin_a)));
    }
}

static ddouble sinq(ddouble a)
{
    /* Strategy.  To compute sin(x), we choose integers a, b so that
     *
     *   x = s + a * (pi/2) + b * (pi/16)
     *
     * and |s| <= pi/32.  Using the fact that
     *
     *   sin(pi/16) = 0.5 * sqrt(2 - sqrt(2 + sqrt(2)))
     *
     * we can compute sin(x) from sin(s), cos(s).  This greatly
     * increases the convergence of the sine Taylor series.
     */
    if (iszeroq(a))
        return Q_ZERO;

    // approximately reduce modulo 2*pi
    ddouble z = roundq(divqq(a, Q_2PI));
    ddouble r = subqq(a, mulqq(Q_2PI, z));

    // approximately reduce modulo pi/2 and then modulo pi/16.
    ddouble t;
    double q = floor(r.hi / Q_PI_2.hi + 0.5);
    t = subqq(r, mulqd(Q_PI_2, q));
    int j = (int)q;
    q = floor(t.hi / _pi_16.hi + 0.5);
    t = subqq(t, mulqd(_pi_16, q));
    int k = (int)q;
    int abs_k = abs(k);

    if (j < -2 || j > 2)
        return nanq();

    if (abs_k > 4)
        return nanq();

    if (k == 0) {
        switch (j)
        {
        case 0:
            return sin_taylor(t);
        case 1:
            return cos_taylor(t);
        case -1:
            return negq(cos_taylor(t));
        default:
            return negq(sin_taylor(t));
        }
    }

    ddouble u = _cos_table[abs_k - 1];
    ddouble v = _sin_table[abs_k - 1];
    ddouble sin_x, cos_x;
    sincos_taylor(t, &sin_x, &cos_x);
    if (j == 0) {
        if (k > 0)
            r = addqq(mulqq(u, sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(u, sin_x), mulqq(v, cos_x));
    } else if (j == 1) {
        if (k > 0)
            r = subqq(mulqq(u, cos_x), mulqq(v, sin_x));
        else
            r = addqq(mulqq(u, cos_x), mulqq(v, sin_x));
    } else if (j == -1) {
        if (k > 0)
            r = subqq(mulqq(v, sin_x), mulqq(u, cos_x));
        else if (k < 0)   /* NOTE! */
            r = subqq(mulqq(negq(u), cos_x), mulqq(v, sin_x));
    } else {
        if (k > 0)
            r = subqq(mulqq(negq(u), sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(v, cos_x), mulqq(u, sin_x));
    }
    return r;
}
ULOOP_UNARY(u_sinq, sinq, ddouble, ddouble)

static ddouble cosq(ddouble a)
{
    if (iszeroq(a))
        return Q_ONE;

    // approximately reduce modulo 2*pi
    ddouble z = roundq(divqq(a, Q_2PI));
    ddouble r = subqq(a, mulqq(Q_2PI, z));

    // approximately reduce modulo pi/2 and then modulo pi/16.
    ddouble t;
    double q = floor(r.hi / Q_PI_2.hi + 0.5);
    t = subqq(r, mulqd(Q_PI_2, q));
    int j = (int)q;
    q = floor(t.hi / _pi_16.hi + 0.5);
    t = subqq(t, mulqd(_pi_16, q));
    int k = (int)q;
    int abs_k = abs(k);

    if (j < -2 || j > 2)
        return nanq();

    if (abs_k > 4)
        return nanq();

    if (k == 0) {
        switch (j) {
        case 0:
            return cos_taylor(t);
        case 1:
            return negq(sin_taylor(t));
        case -1:
            return sin_taylor(t);
        default:
            return negq(cos_taylor(t));
        }
    }

    ddouble sin_x, cos_x;
    sincos_taylor(t, &sin_x, &cos_x);
    ddouble u = _cos_table[abs_k - 1];
    ddouble v = _sin_table[abs_k - 1];

    if (j == 0) {
        if (k > 0)
            r = subqq(mulqq(u, cos_x), mulqq(v, sin_x));
        else
            r = addqq(mulqq(u, cos_x), mulqq(v, sin_x));
    } else if (j == 1) {
        if (k > 0)
            r = subqq(mulqq(negq(u), sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(v, cos_x), mulqq(u, sin_x));
    } else if (j == -1) {
        if (k > 0)
            r = addqq(mulqq(u, sin_x), mulqq(v, cos_x));
        else
            r = subqq(mulqq(u, sin_x), mulqq(v, cos_x));
    } else {
        if (k > 0)
            r = subqq(mulqq(v, sin_x), mulqq(u, cos_x));
        else
            r = subqq(mulqq(negq(u), cos_x), mulqq(v, sin_x));
    }
    return r;
}
ULOOP_UNARY(u_cosq, cosq, ddouble, ddouble)

static ddouble sinhq(ddouble a)
{
    if (iszeroq(a))
        return Q_ZERO;

    if (absq(a).hi > 0.05) {
        ddouble ea = expq(a);
        return mul_pwr2(subqq(ea, reciprocalq(ea)), 0.5);
    }

    /* since a is small, using the above formula gives
     * a lot of cancellation.  So use Taylor series.
     */
    ddouble s = a;
    ddouble t = a;
    ddouble r = sqrq(t);
    double m = 1.0;
    double thresh = fabs((a.hi) * Q_EPS.hi);

    do {
        m += 2.0;
        t = mulqq(t, r);
        t = divqd(t, (m - 1) * m);
        s = addqq(s, t);
    } while (absq(t).hi > thresh);
    return s;
}
ULOOP_UNARY(u_sinhq, sinhq, ddouble, ddouble)

static ddouble coshq(ddouble a)
{
    if (iszeroq(a))
        return Q_ONE;

    ddouble ea = expq(a);
    return mul_pwr2(addqq(ea, reciprocalq(ea)), 0.5);
}
ULOOP_UNARY(u_coshq, coshq, ddouble, ddouble)

static ddouble tanhq(ddouble a)
{
    if (iszeroq(a))
        return Q_ZERO;

    if (fabs(a.hi) > 0.05) {
        ddouble ea = expq(a);
        ddouble inv_ea = reciprocalq(ea);
        return divqq(subqq(ea, inv_ea), addqq(ea, inv_ea));
    }

    ddouble s, c;
    s = sinhq(a);
    c = sqrtq(adddq(1.0, sqrq(s)));
    return divqq(s, c);
}
ULOOP_UNARY(u_tanhq, tanhq, ddouble, ddouble)

/************************* Binary functions ************************/

ULOOP_BINARY(u_hypotqq, hypotqq, ddouble, ddouble, ddouble)
ULOOP_BINARY(u_hypotdq, hypotdq, ddouble, double, ddouble)
ULOOP_BINARY(u_hypotqd, hypotqd, ddouble, ddouble, double)

/************************ Linear algebra ***************************/

static void matmulq(char **args, const npy_intp *dims, const npy_intp* steps,
                    void *data)
{
    // signature (n;i,j),(n;j,k)->(n;i,k)
    const npy_intp nn = dims[0], ii = dims[1], jj = dims[2], kk = dims[3];
    const npy_intp san = steps[0], sbn = steps[1], scn = steps[2],
                   sai = steps[3], saj = steps[4], sbj = steps[5],
                   sbk = steps[6], sci = steps[7], sck = steps[8];
    char *_a = args[0], *_b = args[1], *_c = args[2];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn, _c += scn) {
        for (npy_intp i = 0; i != ii; ++i) {
            for (npy_intp k = 0; k != kk; ++k) {
                ddouble val = Q_ZERO;
                for (npy_intp j = 0; j != jj; ++j) {
                    const ddouble *a_ij =
                            (const ddouble *) (_a + i * sai + j * saj);
                    const ddouble *b_jk =
                            (const ddouble *) (_b + j * sbj + k * sbk);
                    val = addqq(val, mulqq(*a_ij, *b_jk));

                }
                ddouble *c_ik = (ddouble *) (_c + i * sci + k * sck);
                *c_ik = val;
            }
        }
    }
    MARK_UNUSED(data);
}

/*************************** More complicated ***********************/

static void givensq(ddouble f, ddouble g, ddouble *c, ddouble *s, ddouble *r)
{
    /* ACM Trans. Math. Softw. 28(2), 206, Alg 1 */
    if (iszeroq(g)) {
        *c = Q_ONE;
        *s = Q_ZERO;
        *r = f;
    } else if (iszeroq(f)) {
        *c = Q_ZERO;
        *s = (ddouble) {signbitq(g), 0.0};
        *r = absq(g);
    } else {
        *r = copysignqq(hypotqq(f, g), f);

        /* This may come at a slight loss of precision, however, we should
         * not really have to care ...
         */
        ddouble inv_r = reciprocalq(*r);
        *c = mulqq(f, inv_r);
        *s = mulqq(g, inv_r);
    }
}

static void u_givensq(char **args, const npy_intp *dims, const npy_intp* steps,
                      void *data)
{
    // signature (n;2)->(n;2),(n;2,2)
    const npy_intp nn = dims[0];
    const npy_intp san = steps[0], sbn = steps[1], scn = steps[2],
                   sai = steps[3], sbi = steps[4], sci = steps[5],
                   scj = steps[6];
    char *_a = args[0], *_b = args[1], *_c = args[2];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn, _c += scn) {
        ddouble f = *(ddouble *) _a;
        ddouble g = *(ddouble *) (_a + sai);

        ddouble c, s, r;
        givensq(f, g, &c, &s, &r);

        *(ddouble *)_b = r;
        *(ddouble *)(_b + sbi) = Q_ZERO;
        *(ddouble *)_c = c;
        *(ddouble *)(_c + scj) = s;
        *(ddouble *)(_c + sci) = negq(s);
        *(ddouble *)(_c + sci + scj) = c;
    }
    MARK_UNUSED(data);
}

static void svd_tri2x2(ddouble f, ddouble g, ddouble h, ddouble *smin,
                       ddouble *smax, ddouble *cv, ddouble *sv, ddouble *cu,
                       ddouble *su)
{
    ddouble fa = absq(f);
    ddouble ga = absq(g);
    ddouble ha = absq(h);
    bool compute_uv = cv != NULL;

    if (lessqq(fa, ha)) {
        // switch h <-> f, cu <-> sv, cv <-> su
        svd_tri2x2(h, g, f, smin, smax, su, cu, sv, cv);
        return;
    }
    if (iszeroq(ga)) {
        // already diagonal
        *smin = ha;
        *smax = fa;
        if (compute_uv) {
            *cu = Q_ONE;
            *su = Q_ZERO;
            *cv = Q_ONE;
            *sv = Q_ZERO;
        }
        return;
    }
    if (fa.hi < Q_EPS.hi * ga.hi) {
        // ga is very large
        *smax = ga;
        if (ha.hi > 1.0)
            *smin = divqq(fa, divqq(ga, ha));
        else
            *smin = mulqq(divqq(fa, ga), ha);
        if (compute_uv) {
            *cu = Q_ONE;
            *su = divqq(h, g);
            *cv = Q_ONE;
            *sv = divqq(f, g);
        }
        return;
    }
    // normal case
    ddouble fmh = subqq(fa, ha);
    ddouble d = divqq(fmh, fa);
    ddouble q = divqq(g, f);
    ddouble s = subdq(2.0, d);
    ddouble spq = hypotqq(q, s);
    ddouble dpq = hypotqq(d, q);
    ddouble a = mul_pwr2(addqq(spq, dpq), 0.5);
    *smin = absq(divqq(ha, a));
    *smax = absq(mulqq(fa, a));

    if (compute_uv) {
        ddouble tmp = addqq(divqq(q, addqq(spq, s)),
                            divqq(q, addqq(dpq, d)));
        tmp = mulqq(tmp, adddq(1.0, a));
        ddouble tt = hypotqd(tmp, 2.0);
        *cv = divdq(2.0, tt);
        *sv = divqq(tmp, tt);
        *cu = divqq(addqq(*cv, mulqq(*sv, q)), a);
        *su = divqq(mulqq(divqq(h, f), *sv), a);
    }
}

static void u_svd_tri2x2(char **args, const npy_intp *dims,
                         const npy_intp* steps, void *data)
{
    // signature (n;2,2)->(n;2,2),(n;2),(n;2,2)
    const npy_intp nn = dims[0];
    const npy_intp san = steps[0], sbn = steps[1], scn = steps[2],
                   sdn = steps[3], sai = steps[4], saj = steps[5],
                   sbi = steps[6], sbj = steps[7], sci = steps[8],
                   sdi = steps[9], sdj = steps[10];
    char *_a = args[0], *_b = args[1], *_c = args[2], *_d = args[3];

    for (npy_intp n = 0; n != nn;
                ++n, _a += san, _b += sbn, _c += scn, _d += sdn) {
        ddouble f = *(ddouble *) _a;
        ddouble z = *(ddouble *) (_a + sai);
        ddouble g = *(ddouble *) (_a + saj);
        ddouble h = *(ddouble *) (_a + sai + saj);

        ddouble smin, smax, cu, su, cv, sv;
        if (!iszeroq(z)) {
            fprintf(stderr, "svd_tri2x2: matrix is not upper triagonal\n");
            smin = smax = cu = su = cv = sv = nanq();
        } else {
            svd_tri2x2(f, g, h, &smin, &smax, &cv, &sv, &cu, &su);
        }

        *(ddouble *)_b = cu;
        *(ddouble *)(_b + sbj) = negq(su);
        *(ddouble *)(_b + sbi) = su;
        *(ddouble *)(_b + sbi + sbj) = cu;

        *(ddouble *)_c = smax;
        *(ddouble *)(_c + sci) = smin;

        *(ddouble *)_d = cv;
        *(ddouble *)(_d + sdj) = sv;
        *(ddouble *)(_d + sdi) = negq(sv);
        *(ddouble *)(_d + sdi + sdj) = cv;
    }
    MARK_UNUSED(data);
}

static void u_svvals_tri2x2(char **args, const npy_intp *dims,
                            const npy_intp* steps, void *data)
{
    // signature (n;2,2)->(n;2)
    const npy_intp nn = dims[0];
    const npy_intp san = steps[0], sbn = steps[1], sai = steps[2],
                   saj = steps[3], sbi = steps[4];
    char *_a = args[0], *_b = args[1];

    for (npy_intp n = 0; n != nn; ++n, _a += san, _b += sbn) {
        ddouble f = *(ddouble *) _a;
        ddouble z = *(ddouble *) (_a + sai);
        ddouble g = *(ddouble *) (_a + saj);
        ddouble h = *(ddouble *) (_a + sai + saj);

        ddouble smin, smax;
        if (!iszeroq(z)) {
            fprintf(stderr, "svd_tri2x2: matrix is not upper triagonal\n");
            smin = smax = nanq();
        } else {
            svd_tri2x2(f, g, h, &smin, &smax, NULL, NULL, NULL, NULL);
        }

        *(ddouble *)_b = smax;
        *(ddouble *)(_b + sbi) = smin;
    }
    MARK_UNUSED(data);
}

/* ----------------------- Python stuff -------------------------- */

static const char DDOUBLE_WRAP = NPY_CDOUBLE;

static void binary_ufunc(PyObject *module_dict, PyUFuncGenericFunction dq_func,
        PyUFuncGenericFunction qd_func, PyUFuncGenericFunction qq_func,
        char ret_dtype, const char *name, const char *docstring)
{

    PyObject *ufunc;
    PyUFuncGenericFunction* loops = PyMem_New(PyUFuncGenericFunction, 3);
    char *dtypes = PyMem_New(char, 3 * 3);
    void **data = PyMem_New(void *, 3);

    loops[0] = dq_func;
    data[0] = NULL;
    dtypes[0] = NPY_DOUBLE;
    dtypes[1] = DDOUBLE_WRAP;
    dtypes[2] = ret_dtype;

    loops[1] = qd_func;
    data[1] = NULL;
    dtypes[3] = DDOUBLE_WRAP;
    dtypes[4] = NPY_DOUBLE;
    dtypes[5] = ret_dtype;

    loops[2] = qq_func;
    data[2] = NULL;
    dtypes[6] = DDOUBLE_WRAP;
    dtypes[7] = DDOUBLE_WRAP;
    dtypes[8] = ret_dtype;

    ufunc = PyUFunc_FromFuncAndData(
                loops, data, dtypes, 3, 2, 1, PyUFunc_None, name, docstring, 0);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

static void gufunc(PyObject *module_dict, PyUFuncGenericFunction uloop,
                   int nin, int nout, const char *signature, const char *name,
                   const char *docstring)
{
    PyObject *ufunc;
    PyUFuncGenericFunction* loops = PyMem_New(PyUFuncGenericFunction, 1);
    char *dtypes = PyMem_New(char, nin + nout);
    void **data = PyMem_New(void *, 1);

    loops[0] = uloop;
    data[0] = NULL;
    for (int i = 0; i != nin + nout; ++i)
        dtypes[i] = DDOUBLE_WRAP;

    ufunc = PyUFunc_FromFuncAndDataAndSignature(
                loops, data, dtypes, 1, nin, nout, PyUFunc_None, name,
                docstring, 0, signature);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

static void unary_ufunc(PyObject *module_dict,
                        PyUFuncGenericFunction func, char ret_dtype,
                        const char *name, const char *docstring)
{
    PyObject *ufunc;
    PyUFuncGenericFunction* loops = PyMem_New(PyUFuncGenericFunction, 1);
    char *dtypes = PyMem_New(char, 1 * 2);
    void **data = PyMem_New(void *, 1);

    loops[0] = func;
    data[0] = NULL;
    dtypes[0] = DDOUBLE_WRAP;
    dtypes[1] = ret_dtype;

    ufunc = PyUFunc_FromFuncAndData(
                loops, data, dtypes, 1, 1, 1, PyUFunc_None, name, docstring, 0);
    PyDict_SetItemString(module_dict, name, ufunc);
    Py_DECREF(ufunc);
}

static void constant(PyObject *module_dict, ddouble value, const char *name)
{
    // Note that data must be allocated using malloc, not python allocators!
    ddouble *data = malloc(sizeof value);
    *data = value;

    PyArrayObject *array = (PyArrayObject *)
        PyArray_SimpleNewFromData(0, NULL, DDOUBLE_WRAP, data);
    PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS(array, NPY_ARRAY_WRITEABLE);

    PyDict_SetItemString(module_dict, name, (PyObject *)array);
    Py_DECREF(array);
}

PyMODINIT_FUNC PyInit__raw(void)
{
    // Defitions
    static PyMethodDef no_methods[] = {
        {NULL, NULL, 0, NULL}    // No methods defined
    };
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_raw",
        NULL,
        -1,
        no_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    /* Module definition */
    PyObject *module, *module_dict;
    PyArray_Descr *dtype;

    /* Create module */
    module = PyModule_Create(&module_def);
    if (!module)
        return NULL;
    module_dict = PyModule_GetDict(module);

    /* Initialize numpy things */
    import_array();
    import_umath();

    /* Create ufuncs */
    binary_ufunc(module_dict, u_adddq, u_addqd, u_addqq,
                 DDOUBLE_WRAP, "add", "addition");
    binary_ufunc(module_dict, u_subdq, u_subqd, u_subqq,
                 DDOUBLE_WRAP, "subtract", "subtraction");
    binary_ufunc(module_dict, u_muldq, u_mulqd, u_mulqq,
                 DDOUBLE_WRAP, "multiply", "element-wise multiplication");
    binary_ufunc(module_dict, u_divdq, u_divqd, u_divqq,
                 DDOUBLE_WRAP, "true_divide", "element-wise division");

    binary_ufunc(module_dict, u_equaldq, u_equalqd, u_equalqq,
                 NPY_BOOL, "equal", "equality comparison");
    binary_ufunc(module_dict, u_notequaldq, u_notequalqd, u_notequalqq,
                 NPY_BOOL, "not_equal", "inequality comparison");
    binary_ufunc(module_dict, u_greaterdq, u_greaterqd, u_greaterqq,
                 NPY_BOOL, "greater", "element-wise greater");
    binary_ufunc(module_dict, u_lessdq, u_lessqd, u_lessqq,
                 NPY_BOOL, "less", "element-wise less");
    binary_ufunc(module_dict, u_greaterequaldq, u_greaterequalqd, u_greaterequalqq,
                 NPY_BOOL, "greater_equal", "element-wise greater or equal");
    binary_ufunc(module_dict, u_lessequaldq, u_lessequalqd, u_lessequalqq,
                 NPY_BOOL, "less_equal", "element-wise less or equal");
    binary_ufunc(module_dict, u_fmindq, u_fminqd, u_fminqq,
                 DDOUBLE_WRAP, "fmin", "element-wise minimum");
    binary_ufunc(module_dict, u_fmaxdq, u_fmaxqd, u_fmaxqq,
                 DDOUBLE_WRAP, "fmax", "element-wise minimum");

    unary_ufunc(module_dict, u_negq, DDOUBLE_WRAP,
                "negative", "negation (+ to -)");
    unary_ufunc(module_dict, u_posq, DDOUBLE_WRAP,
                "positive", "explicit + sign");
    unary_ufunc(module_dict, u_absq, DDOUBLE_WRAP,
                "absolute", "absolute value");
    unary_ufunc(module_dict, u_reciprocalq, DDOUBLE_WRAP,
                "reciprocal", "element-wise reciprocal value");
    unary_ufunc(module_dict, u_sqrq, DDOUBLE_WRAP,
                "square", "element-wise square");
    unary_ufunc(module_dict, u_sqrtq, DDOUBLE_WRAP,
                "sqrt", "element-wise square root");
    unary_ufunc(module_dict, u_signbitq, NPY_BOOL,
                "signbit", "sign bit of number");
    unary_ufunc(module_dict, u_isfiniteq, NPY_BOOL,
                "isfinite", "whether number is finite");
    unary_ufunc(module_dict, u_isinfq, NPY_BOOL,
                "isinf", "whether number is infinity");
    unary_ufunc(module_dict, u_isnanq, NPY_BOOL,
                "isnan", "test for not-a-number");

    unary_ufunc(module_dict, u_roundq, DDOUBLE_WRAP,
                "rint", "round to nearest integer");
    unary_ufunc(module_dict, u_floorq, DDOUBLE_WRAP,
                "floor", "round down to next integer");
    unary_ufunc(module_dict, u_ceilq, DDOUBLE_WRAP,
                "ceil", "round up to next integer");
    unary_ufunc(module_dict, u_expq, DDOUBLE_WRAP,
                "exp", "exponential function");
    unary_ufunc(module_dict, u_expm1q, DDOUBLE_WRAP,
                "expm1", "exponential function minus one");
    unary_ufunc(module_dict, u_logq, DDOUBLE_WRAP,
                "log", "natural logarithm");
    unary_ufunc(module_dict, u_sinq, DDOUBLE_WRAP,
                "sin", "sine");
    unary_ufunc(module_dict, u_cosq, DDOUBLE_WRAP,
                "cos", "cosine");
    unary_ufunc(module_dict, u_sinhq, DDOUBLE_WRAP,
                "sinh", "hyperbolic sine");
    unary_ufunc(module_dict, u_coshq, DDOUBLE_WRAP,
                "cosh", "hyperbolic cosine");
    unary_ufunc(module_dict, u_tanhq, DDOUBLE_WRAP,
                "tanh", "hyperbolic tangent");

    unary_ufunc(module_dict, u_iszeroq, NPY_BOOL,
                "iszero", "element-wise test for zero");
    unary_ufunc(module_dict, u_isoneq, NPY_BOOL,
                "isone", "element-wise test for one");
    unary_ufunc(module_dict, u_ispositiveq, NPY_BOOL,
                "ispositive", "element-wise test for positive values");
    unary_ufunc(module_dict, u_isnegativeq, NPY_BOOL,
                "isnegative", "element-wise test for negative values");
    unary_ufunc(module_dict, u_signq, DDOUBLE_WRAP,
                "sign", "element-wise sign computation");

    binary_ufunc(module_dict, u_copysigndq, u_copysignqd, u_copysignqq,
                 DDOUBLE_WRAP, "copysign", "overrides sign of x with that of y");
    binary_ufunc(module_dict, u_hypotdq, u_hypotqd, u_hypotqq,
                 DDOUBLE_WRAP, "hypot", "hypothenuse calculation");


    constant(module_dict, Q_MAX, "MAX");
    constant(module_dict, Q_MIN, "MIN");
    constant(module_dict, Q_EPS, "EPS");
    constant(module_dict, Q_2PI, "TWOPI");
    constant(module_dict, Q_PI, "PI");
    constant(module_dict, Q_PI_2, "PI_2");
    constant(module_dict, Q_PI_4, "PI_4");
    constant(module_dict, Q_E, "E");
    constant(module_dict, Q_LOG2, "LOG2");
    constant(module_dict, Q_LOG10, "LOG10");

    gufunc(module_dict, matmulq, 2, 1, "(i?,j),(j,k?)->(i?,k?)",
           "matmul", "Matrix multiplication");
    gufunc(module_dict, u_givensq, 1, 2, "(2)->(2),(2,2)",
           "givens", "Generate Givens rotation");
    gufunc(module_dict, u_svd_tri2x2, 1, 3, "(2,2)->(2,2),(2),(2,2)",
           "svd_tri2x2", "SVD of upper triangular 2x2 problem");
    gufunc(module_dict, u_svvals_tri2x2, 1, 1, "(2,2)->(2)",
           "svvals_tri2x2", "singular values of upper triangular 2x2 problem");

    /* Make dtype */
    dtype = PyArray_DescrFromType(DDOUBLE_WRAP);
    PyDict_SetItemString(module_dict, "dtype", (PyObject *)dtype);

    /* Module is ready */
    return module;
}
