import tenseal as ts
import pickle

# QUESTION: 2
context_q2 = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
plain_vector = [138]
encrypted_vector = ts.bfv_vector(context_q2, plain_vector)
result_q2 = encrypted_vector * [914]

# QUESTION: 3
with open('R11922138.pkl', 'rb') as f:
    given = pickle.load(f)
context_q3 = ts.context_from(given['context'])
encrypted_a = ts.bfv_vector_from(context_q3, given['encrypted_a'])
encrypted_b = ts.bfv_vector_from(context_q3, given['encrypted_b'])
result_q3 = encrypted_a * 138 + encrypted_b

# package results
result = {
   'q2_context': context_q2.serialize(save_secret_key=True),
   'q2_result': result_q2.serialize(),
   'q3_result': result_q3.serialize()
}
with open('R11922138_a1.pkl', 'wb') as f:
    pickle.dump(result, f)
# with open('R11922138_a1.pkl', 'rb') as f:
#     given = pickle.load(f)
# context = ts.context_from(given['q2_context'])
# print(ts.bfv_vector_from(context, given['q2_result']).decrypt())
# print(ts.bfv_vector_from(context_q3, given['q3_result']))
