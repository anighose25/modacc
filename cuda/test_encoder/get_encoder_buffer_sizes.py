import sys
batch_size =int(sys.argv[1])
sequence_length=int(sys.argv[2])
hidden_size=int(sys.argv[3])
intermediate_size=int(sys.argv[4])
nh=int(sys.argv[5])
nq=int(sys.argv[6])
dump_script=1
input_buf=(batch_size * sequence_length * hidden_size)
input_norm=(batch_size * sequence_length * hidden_size)
input_mask=(batch_size * sequence_length * sequence_length * nh)
norm_weights=(hidden_size)
norm_bias=(hidden_size)
norm_var=(batch_size * sequence_length)
norm_mean=(batch_size * sequence_length)
qkv_weights=(3 * hidden_size * hidden_size)
qkv_bias=(3 * hidden_size)
qkv_out=(3 * hidden_size * batch_size * sequence_length)
keys=(batch_size * sequence_length * hidden_size)
queries=(batch_size * sequence_length * hidden_size)
values=(batch_size * sequence_length * hidden_size)
soft_out=(batch_size * sequence_length * sequence_length * nh)
context=(batch_size * sequence_length * sequence_length * nh)
attn_prob_dropout_mask=(batch_size * sequence_length * sequence_length * nh)
buf_1=(batch_size * sequence_length * hidden_size)
attn_o_inp=(batch_size * sequence_length * hidden_size)
attn_ow=(batch_size * hidden_size * hidden_size)
add_res=(batch_size * sequence_length * hidden_size)
attn_ob=(hidden_size)
attn_output_dropout_mask=(batch_size * sequence_length * hidden_size)
ff1_inp=(batch_size * sequence_length * hidden_size)
attn_nw=(hidden_size)
attn_nb=(hidden_size)
attn_norm_var=(batch_size * sequence_length)
attn_norm_mean=(batch_size * sequence_length)
inter_w=(hidden_size * intermediate_size)
inter_b=(intermediate_size)
ff2_inp=(batch_size * sequence_length * intermediate_size)
output_w=(intermediate_size * hidden_size)
output_b=(hidden_size)
out=(batch_size * sequence_length * hidden_size)
layer_output_dropout_mask=(batch_size * sequence_length * hidden_size)

if dump_script!=1:
	print "input_buf",input_buf
	print "input_norm",input_norm
	print "input_mask",input_mask
	print "norm_weights",norm_weights
	print "norm_bias",norm_bias
	print "norm_var",norm_var
	print "norm_mean",norm_mean
	print "qkv_weights",qkv_weights
	print "qkv_bias",qkv_bias
	print "qkv_out",qkv_out
	print "keys",keys
	print "queries",queries
	print "values",values
	print "soft_out",soft_out
	print "context",context
	print "attn_prob_dropout_mask",attn_prob_dropout_mask
	print "buf_1",buf_1
	print "attn_o_inp",attn_o_inp
	print "attn_ow",attn_ow
	print "add_res",add_res
	print "attn_ob",attn_ob
	print "attn_output_dropout_mask",attn_output_dropout_mask
	print "ff1_inp",ff1_inp
	print "attn_nw",attn_nw
	print "attn_nb",attn_nb
	print "attn_norm_var",attn_norm_var
	print "attn_norm_mean",attn_norm_mean
	print "inter_w",inter_w
	print "inter_b",inter_b
	print "ff2_inp",ff2_inp
	print "output_w",output_w
	print "output_b",output_b
	print "out",out
	print "layer_output_dropout_mask",layer_output_dropout_mask
else:

	print "./logGP "+str(input_buf)+" 1 >> logGPdumps/buffer_stats_input_buf.txt"
	print "./logGP "+str(input_norm)+" 1 >> logGPdumps/buffer_stats_input_norm.txt"
	print "./logGP "+str(input_mask)+" 1 >> logGPdumps/buffer_stats_input_mask.txt"
	print "./logGP "+str(norm_weights)+" 1 >> logGPdumps/buffer_stats_norm_weights.txt"
	print "./logGP "+str(norm_bias)+" 1 >> logGPdumps/buffer_stats_norm_bias.txt"
	print "./logGP "+str(norm_var)+" 1 >> logGPdumps/buffer_stats_norm_var.txt"
	print "./logGP "+str(norm_mean)+" 1 >> logGPdumps/buffer_stats_norm_mean.txt"
	print "./logGP "+str(qkv_weights)+" 1 >> logGPdumps/buffer_stats_qkv_weights.txt"
	print "./logGP "+str(qkv_bias)+" 1 >> logGPdumps/buffer_stats_qkv_bias.txt"
	print "./logGP "+str(qkv_out)+" 1 >> logGPdumps/buffer_stats_qkv_out.txt"
	print "./logGP "+str(keys)+" 1 >> logGPdumps/buffer_stats_keys.txt"
	print "./logGP "+str(queries)+" 1 >> logGPdumps/buffer_stats_queries.txt"
	print "./logGP "+str(values)+" 1 >> logGPdumps/buffer_stats_values.txt"
	print "./logGP "+str(soft_out)+" 1 >> logGPdumps/buffer_stats_soft_out.txt"
	print "./logGP "+str(context)+" 1 >> logGPdumps/buffer_stats_context.txt"
	print "./logGP "+str(attn_prob_dropout_mask)+" 1 >> logGPdumps/buffer_stats_attn_prob_dropout_mask.txt"
	print "./logGP "+str(buf_1)+" 1 >> logGPdumps/buffer_stats_buf_1.txt"
	print "./logGP "+str(attn_o_inp)+" 1 >> logGPdumps/buffer_stats_attn_o_inp.txt"
	print "./logGP "+str(attn_ow)+" 1 >> logGPdumps/buffer_stats_attn_ow.txt"
	print "./logGP "+str(add_res)+" 1 >> logGPdumps/buffer_stats_add_res.txt"
	print "./logGP "+str(attn_ob)+" 1 >> logGPdumps/buffer_stats_attn_ob.txt"
	print "./logGP "+str(attn_output_dropout_mask)+" 1 >> logGPdumps/buffer_stats_attn_output_dropout_mask.txt"
	print "./logGP "+str(ff1_inp)+" 1 >> logGPdumps/buffer_stats_ff1_inp.txt"
	print "./logGP "+str(attn_nw)+" 1 >> logGPdumps/buffer_stats_attn_nw.txt"
	print "./logGP "+str(attn_nb)+" 1 >> logGPdumps/buffer_stats_attn_nb.txt"
	print "./logGP "+str(attn_norm_var)+" 1 >> logGPdumps/buffer_stats_attn_norm_var.txt"
	print "./logGP "+str(attn_norm_mean)+" 1 >> logGPdumps/buffer_stats_attn_norm_mean.txt"
	print "./logGP "+str(inter_w)+" 1 >> logGPdumps/buffer_stats_inter_w.txt"
	print "./logGP "+str(inter_b)+" 1 >> logGPdumps/buffer_stats_inter_b.txt"
	print "./logGP "+str(ff2_inp)+" 1 >> logGPdumps/buffer_stats_ff2_inp.txt"
	print "./logGP "+str(output_w)+" 1 >> logGPdumps/buffer_stats_output_w.txt"
	print "./logGP "+str(output_b)+" 1 >> logGPdumps/buffer_stats_output_b.txt"
	print "./logGP "+str(out)+" 1 >> logGPdumps/buffer_stats_out.txt"
	print "./logGP "+str(layer_output_dropout_mask)+" 1 >> logGPdumps/buffer_stats_layer_output_dropout_mask.txt"
