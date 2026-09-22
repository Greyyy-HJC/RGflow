#include <cstdint>
#include <complex>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <comm_quda.h>
#include <gauge_field.h>
#include <gauge_path_quda.h>
#include <gauge_tools.h>
#include <pgauge_monte.h>
#include <quda.h>
#include <quda_internal.h>
#include <quda_ptr.h>
#include <random_quda.h>
#include <unitarization_links.h>

namespace {

thread_local std::string last_error;

void set_reunitarization_constants()
{
  quda::setUnitarizeLinksConstants(1e-14, 1e-10, 1, 0, 1e-6, 1e-6);
}

QudaGaugeParam gauge_param(const int *dims)
{
  QudaGaugeParam param = newQudaGaugeParam();
  for (int direction = 0; direction < 4; ++direction) param.X[direction] = dims[direction];
  param.location = QUDA_CPU_FIELD_LOCATION;
  param.cpu_prec = QUDA_DOUBLE_PRECISION;
  param.cuda_prec = QUDA_SINGLE_PRECISION;
  param.cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  param.reconstruct = QUDA_RECONSTRUCT_NO;
  param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  param.type = QUDA_WILSON_LINKS;
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;
  param.anisotropy = 1.0;
  param.tadpole_coeff = 1.0;
  return param;
}

struct HeatbathContext {
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  quda::GaugeField gauge = {};
  quda::GaugeField gauge_extended = {};
  quda::RNG rng = {};

  HeatbathContext(const int *dims, std::uint64_t seed, bool cold_start)
  {
    gauge_param = ::gauge_param(dims);

    quda::GaugeFieldParam gauge_field_param(gauge_param);
    gauge_field_param.location = QUDA_CUDA_FIELD_LOCATION;
    gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gauge_field_param.create = QUDA_NULL_FIELD_CREATE;
    gauge_field_param.link_type = QUDA_WILSON_LINKS;
    gauge_field_param.reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_field_param.setPrecision(QUDA_SINGLE_PRECISION, true);
    gauge = quda::GaugeField(gauge_field_param);

    quda::lat_dim_t extended_dims;
    quda::lat_dim_t radius = {0, 0, 0, 0};
    for (int direction = 0; direction < 4; ++direction) {
      if (quda::comm_dim_partitioned(direction)) radius[direction] = 2;
      extended_dims[direction] = dims[direction] + 2 * radius[direction];
    }

    quda::GaugeFieldParam extended_param(extended_dims, QUDA_SINGLE_PRECISION, QUDA_RECONSTRUCT_NO, 0,
                                         QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
    extended_param.create = QUDA_ZERO_FIELD_CREATE;
    extended_param.location = QUDA_CUDA_FIELD_LOCATION;
    extended_param.order = gauge_field_param.order;
    extended_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    extended_param.t_boundary = QUDA_PERIODIC_T;
    extended_param.nFace = 1;
    for (int direction = 0; direction < 4; ++direction) extended_param.r[direction] = radius[direction];
    gauge_extended = quda::GaugeField(extended_param);

    rng = quda::RNG(gauge_extended, seed);
    if (cold_start)
      quda::InitGaugeField(gauge_extended);
    else
      quda::InitGaugeField(gauge_extended, rng);
    set_reunitarization_constants();
  }

  ~HeatbathContext() { quda::PGaugeExchangeFree(); }

  void update(double beta, int heatbath_hits, int overrelaxation_hits, int steps)
  {
    quda::quda_ptr failures(QUDA_MEMORY_HOST_PINNED, sizeof(int), false);
    int &host_failures = *static_cast<int *>(failures.data_host());
    int &device_failures = *static_cast<int *>(failures.data_device());

    for (int step = 0; step < steps; ++step) {
      host_failures = 0;
      quda::Monte(gauge_extended, rng, beta, heatbath_hits, overrelaxation_hits);
      quda::unitarizeLinks(gauge_extended, &device_failures);
      if (host_failures != 0) throw std::runtime_error("QUDA failed to reunitarize gauge links");
    }
  }

  void copy_to_host(void **host_links)
  {
    quda::copyExtendedGauge(gauge, gauge_extended, QUDA_CUDA_FIELD_LOCATION);
    quda::GaugeFieldParam host_param(gauge_param, host_links);
    host_param.location = QUDA_CPU_FIELD_LOCATION;
    host_param.order = QUDA_QDP_GAUGE_ORDER;
    host_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    host_param.create = QUDA_REFERENCE_FIELD_CREATE;
    quda::GaugeField host_gauge(host_param);
    host_gauge = gauge;
  }
};

template <typename Function> int run_checked(Function function)
{
  try {
    function();
    last_error.clear();
    return 0;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 1;
  }
}

} // namespace

extern "C" {

void *rgflow_heatbath_create(const int *dims, std::uint64_t seed, int cold_start)
{
  try {
    auto *context = new HeatbathContext(dims, seed, cold_start != 0);
    last_error.clear();
    return context;
  } catch (const std::exception &error) {
    last_error = error.what();
    return nullptr;
  }
}

int rgflow_heatbath_update(void *context, double beta, int heatbath_hits, int overrelaxation_hits, int steps)
{
  return run_checked([&] {
    static_cast<HeatbathContext *>(context)->update(beta, heatbath_hits, overrelaxation_hits, steps);
  });
}

int rgflow_heatbath_copy_to_host(void *context, void **host_links)
{
  return run_checked([&] { static_cast<HeatbathContext *>(context)->copy_to_host(host_links); });
}

int rgflow_gauge_loop_trace(const int *dims, void **host_links, const int *paths, const int *path_lengths,
                            int num_paths, int max_length, double *trace_real, double *trace_imag)
{
  return run_checked([&] {
    QudaGaugeParam param = gauge_param(dims);
    quda::GaugeFieldParam host_param(param, host_links);
    host_param.location = QUDA_CPU_FIELD_LOCATION;
    host_param.order = QUDA_QDP_GAUGE_ORDER;
    host_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    host_param.create = QUDA_REFERENCE_FIELD_CREATE;
    quda::GaugeField host_gauge(host_param);

    quda::GaugeFieldParam device_param(param);
    device_param.location = QUDA_CUDA_FIELD_LOCATION;
    device_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    device_param.create = QUDA_NULL_FIELD_CREATE;
    device_param.link_type = QUDA_WILSON_LINKS;
    device_param.reconstruct = QUDA_RECONSTRUCT_NO;
    device_param.setPrecision(QUDA_SINGLE_PRECISION, true);
    quda::GaugeField device_gauge(device_param);
    device_gauge = host_gauge;

    std::vector<int *> path_rows(num_paths);
    for (int index = 0; index < num_paths; ++index)
      path_rows[index] = const_cast<int *>(paths + index * max_length);
    std::vector<int **> input_paths = {path_rows.data()};
    std::vector<int> lengths(path_lengths, path_lengths + num_paths);
    std::vector<double> coefficients(num_paths, 1.0);
    std::vector<quda::Complex> traces(num_paths);
    quda::gaugeLoopTrace(device_gauge, traces, 1.0, input_paths, lengths, coefficients, num_paths, max_length);
    for (int index = 0; index < num_paths; ++index) {
      trace_real[index] = traces[index].real();
      trace_imag[index] = traces[index].imag();
    }
  });
}

void rgflow_heatbath_destroy(void *context) { delete static_cast<HeatbathContext *>(context); }

const char *rgflow_heatbath_last_error() { return last_error.c_str(); }

}
