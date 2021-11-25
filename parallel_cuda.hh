/****************************************************************************
 *
 * parallel_sort_matching.hh
 *
 * Parallel Sort-Based matching
 *
 * Copyright (C) 2016, 2018, 2019 Moreno Marzolla, Gabriele D'Angelo
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#ifndef PARALLEL_CUDA
#define PARALLEL_CUDA

#include "interval.h"

#ifdef __cplusplus
extern "C" {
#endif
size_t parallel_cuda( const struct interval* sub, size_t n,
                               const struct interval* upd, size_t m );

#ifdef __cplusplus
}
#endif

#endif /* PARALLEL_CUDA */
