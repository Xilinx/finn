package mvu_pkg;
	function int unsigned mvu_pipeline_depth(
		input string        core,
		input int unsigned  simd   = 0,
		input int unsigned  seglen = 0
	);
		unique case(core)
		"mvu_vvu_8sx9_dsp58": begin
			automatic int  chainlen = (simd+2)/3;
			if(seglen == 0)  seglen = chainlen;
			return  3 + (chainlen-1)/seglen;
		end
		"mvu_4sx4u", "mvu_4sx4u_dsp48e1", "mvu_4sx4u_dsp48e2",
		"mvu_8sx8u_dsp48":
			return  5;
		default: begin
			$error("Unknown MVU core '%s'", core);
			$finish;
		end
		endcase
	endfunction : mvu_pipeline_depth
endpackage : mvu_pkg
