#pragma once

#include "mesh.h"
#include "../utils/utils.h"

Mesh LoadWavefrontObjMesh(const fs::path &path, const MeshSettings &settings);
