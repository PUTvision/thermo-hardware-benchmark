#
# Main component of TF Micro project 'thermo'.
#

idf_component_register(
    SRCS main.cc main_functions.cc model_utils.cc unet_model.cc
    PRIV_REQUIRES tflite-lib spi_flash
    INCLUDE_DIRS "")
