#ifndef ERRORS_H
#define ERRORS_H

namespace MX
{
    namespace Utils
    {
        enum MX_status
        {
            MX_STATUS_OK = 0,
            MX_STATUS_TIMEOUT = 1,
            MX_STATUS_INVALID_DFP,
            MX_STATUS_INVALID_DATA_FMT,
            MX_STATUS_INVALID_CHIP_GEN,
            MX_STATUS_ERR_OPEN_DEV,
            MX_STATUS_ERR_GET_CHIPNUM,
            MX_STATUS_ERR_DFP_MISMATCH_WITH_HARDWARE,
            MX_STATUS_ERR_DOWNLOAD_MODEL,
            MX_STATUS_ERR_ENABLE_STREAM,
            MX_STATUS_ERR_STREAM_IFMAP,
            MX_STATUS_ERR_STREAM_OFMAP,
            MX_STATUS_END,

            MX_STATUS_ERR_INTERNAL = 1000, // error code >= 1000
        };
    } // namespace MX
} // namespace Utils

#endif