Split the affine reconstruction out of the dot product. The codes are multiplied by `x` group by group, and the scale and bias are applied to two small `(d_out, groups)` arrays.
