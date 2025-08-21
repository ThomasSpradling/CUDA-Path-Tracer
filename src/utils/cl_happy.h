#pragma once

// Hack to maintain separable compilation wtihout MSVC or similar complaining
#ifndef __host__
  #define __host__
#endif

#ifndef __device__
  #define __device__
#endif

#ifndef __forceinline__
  #define __forceinline__ inline
#endif
