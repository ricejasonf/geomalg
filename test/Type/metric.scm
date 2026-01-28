; RUN: printenv
; RUN: heavy-scheme -I %heavy_module_path -I %geomalg_module_path %s | FileCheck

(import (heavy base)
        (geomalg base))

