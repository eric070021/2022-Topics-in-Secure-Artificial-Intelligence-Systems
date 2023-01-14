import tenseal as ts

# QUESTION: 4
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60]
          )
context.global_scale = 2**40

v1 = [4]
v2 = [2]

enc_v1 = ts.ckks_vector(context, v1)
enc_v2 = ts.ckks_vector(context, v2)

# multiple adds
origin = 4
result = enc_v1
for i in range(10):
    result = result + enc_v2
    origin = origin + 2
    print("Noise[{}]: {}".format(i+1, abs(origin - result.decrypt()[0])))

print("Adding multiple times: {}".format(result.decrypt())) # should be 24

# multiple times
origin = 4
result = enc_v1
for i in range(10):
    result = result * enc_v2
    origin = origin * 2
    print("Noise[{}]: {}".format(i+1, abs(origin - result.decrypt()[0])))

print("Timing multiple times: {}".format(result.decrypt())) # should be 4096
