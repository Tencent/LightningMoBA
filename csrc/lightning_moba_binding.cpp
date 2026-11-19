#include <pybind11/pybind11.h>

void init_lightning_moba(pybind11::module &);
void init_flash_topk(pybind11::module &);
void init_flash_dynamic(pybind11::module &);
void init_flash_dynamic_attention(pybind11::module &);
void init_flash_kvcentric_stream(pybind11::module &);
void init_block8(pybind11::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "LightningMoBA";
    init_lightning_moba(m);
    init_flash_topk(m);
    init_flash_dynamic(m);
    init_flash_dynamic_attention(m);
    init_flash_kvcentric_stream(m);
    init_block8(m);
}
