"""Generate the f16-output variants of the elementwise kernels.

Activations that only ever feed a matmul do not need to be f32 in memory. The
matmul narrows them to f16 on the way into LDS anyway, so producing them as f16
halves the traffic on both sides of the handoff. The residual stream, the norm
statistics and the final output stay f32.
"""
from pathlib import Path

KERNELS = Path(__file__).resolve().parent.parent / "kernels"

ROPE_VIEWS_F32 = """  %x_view = buffer.view %x_global[%c0_offset] : buffer -> view<[%token_count]x[%row_stride]xf32>"""
ROPE_VIEWS_F16 = """  %x_view = buffer.view %x_global[%c0_offset] : buffer -> view<[%token_count]x[%row_stride]xf16>"""

EDITS = {
    "layernorm_f32.loom": (
        "layernorm_f32_to_f16.loom", "dinov3_layernorm_f32", "dinov3.layernorm_f32",
        [
            ("%output_view = buffer.view %output_noalias[%c0_offset] : buffer -> view<[%token_count]x[%hidden_size]xf32>",
             "%output_view = buffer.view %output_noalias[%c0_offset] : buffer -> view<[%token_count]x[%hidden_size]xf16>"),
            ("""      %result = scalar.addf<reassoc|nnan|ninf|nsz> %scaled, %shift : f32
      view.store %result, %output_view[%token, %channel] : f32, view<[%token_count]x[%hidden_size]xf32>""",
             """      %result = scalar.addf<reassoc|nnan|ninf|nsz> %scaled, %shift : f32
      %narrow = scalar.fptrunc %result : f32 to f16
      view.store %narrow, %output_view[%token, %channel] : f16, view<[%token_count]x[%hidden_size]xf16>"""),
        ],
        "// VARIANT: writes f16. The consumer is a matmul, which would narrow it anyway.",
    ),
    "swiglu_f32.loom": (
        "swiglu_f16.loom", "dinov3_swiglu_f32", "dinov3.swiglu_f32",
        [
            ("""  %gate_view = buffer.view %gate_global[%c0_offset] : buffer -> view<[%token_count]x[%row_stride]xf32>
  %up_view = buffer.view %up_global[%c0_offset] : buffer -> view<[%token_count]x[%row_stride]xf32>
  %out_view = buffer.view %out_global[%c0_offset] : buffer -> view<[%token_count]x[%width]xf32>""",
             """  %gate_view = buffer.view %gate_global[%c0_offset] : buffer -> view<[%token_count]x[%row_stride]xf16>
  %up_view = buffer.view %up_global[%c0_offset] : buffer -> view<[%token_count]x[%row_stride]xf16>
  %out_view = buffer.view %out_global[%c0_offset] : buffer -> view<[%token_count]x[%width]xf16>"""),
            ("""      %gate_value = view.load %gate_view[%token, %channel] : view<[%token_count]x[%row_stride]xf32> -> f32
      %up_value = view.load %up_view[%token, %channel] : view<[%token_count]x[%row_stride]xf32> -> f32
      %activated = scalar.siluf<afn> %gate_value : f32
      %result = scalar.mulf %activated, %up_value : f32
      view.store %result, %out_view[%token, %channel] : f32, view<[%token_count]x[%width]xf32>""",
             """      %gate_half = view.load %gate_view[%token, %channel] : view<[%token_count]x[%row_stride]xf16> -> f16
      %up_half = view.load %up_view[%token, %channel] : view<[%token_count]x[%row_stride]xf16> -> f16
      %gate_value = scalar.extf %gate_half : f16 to f32
      %up_value = scalar.extf %up_half : f16 to f32
      %activated = scalar.siluf<afn> %gate_value : f32
      %result = scalar.mulf %activated, %up_value : f32
      %narrow = scalar.fptrunc %result : f32 to f16
      view.store %narrow, %out_view[%token, %channel] : f16, view<[%token_count]x[%width]xf16>"""),
        ],
        "// VARIANT: f16 in and out. Both sides are matmul handoffs; the silu itself is f32.",
    ),
    "flash_attention_f16_wmma.loom": (
        "flash_attention_f16_wmma_cf16.loom", "dinov3_flash_attention_f16_wmma",
        "dinov3.flash_attention_f16_wmma",
        [
            ("%out_view = buffer.view %out_global[%c0_offset] : buffer -> view<[%token_count]x[%hidden_size]xf32>",
             "%out_view = buffer.view %out_global[%c0_offset] : buffer -> view<[%token_count]x[%hidden_size]xf16>"),
            ("""    %values = vector.load %result_view[%publish_row, %publish_dim] : view<16x212xf32> -> vector<4xf32>""",
             """    %wide = vector.load %result_view[%publish_row, %publish_dim] : view<16x212xf32> -> vector<4xf32>
    %values = vector.fptrunc %wide : vector<4xf32> to vector<4xf16>"""),
            ("""    vector.store %values, %out_view[%bounded, %out_slot] : vector<4xf32>, view<[%token_count]x[%hidden_size]xf32>""",
             """    vector.store %values, %out_view[%bounded, %out_slot] : vector<4xf16>, view<[%token_count]x[%hidden_size]xf16>"""),
        ],
        "// VARIANT: writes f16 straight into the output projection's activation.",
    ),
    "rope_2d_f32.loom": (
        "rope_2d_f16.loom", "dinov3_rope_2d_f32", "dinov3.rope_2d_f32",
        [
            (ROPE_VIEWS_F32, ROPE_VIEWS_F16),
            ("""      %low = view.load %x_view[%token, %low_slot] : view<[%token_count]x[%row_stride]xf32> -> f32
      %high = view.load %x_view[%token, %high_slot] : view<[%token_count]x[%row_stride]xf32> -> f32""",
             """      %low_half = view.load %x_view[%token, %low_slot] : view<[%token_count]x[%row_stride]xf16> -> f16
      %high_half = view.load %x_view[%token, %high_slot] : view<[%token_count]x[%row_stride]xf16> -> f16
      %low = scalar.extf %low_half : f16 to f32
      %high = scalar.extf %high_half : f16 to f32"""),
            ("""      view.store %new_low, %x_view[%token, %low_slot] : f32, view<[%token_count]x[%row_stride]xf32>
      view.store %new_high, %x_view[%token, %high_slot] : f32, view<[%token_count]x[%row_stride]xf32>""",
             """      %narrow_low = scalar.fptrunc %new_low : f32 to f16
      %narrow_high = scalar.fptrunc %new_high : f32 to f16
      view.store %narrow_low, %x_view[%token, %low_slot] : f16, view<[%token_count]x[%row_stride]xf16>
      view.store %narrow_high, %x_view[%token, %high_slot] : f16, view<[%token_count]x[%row_stride]xf16>"""),
        ],
        "// VARIANT: rotates an f16 tensor. The rotation itself is done in f32.",
    ),
    "flash_attention_f16_wmma_cf16.loom": (
        "flash_attention_f16_wmma_af16_cf16.loom", "dinov3_flash_attention_f16_wmma_cf16",
        "dinov3.flash_attention_f16_wmma_cf16",
        [
            ("""  %q_view = buffer.view %q_global[%c0_offset] : buffer -> view<[%token_count]x[%qkv_stride]xf32>
  %k_view = buffer.view %k_global[%c0_offset] : buffer -> view<[%token_count]x[%qkv_stride]xf32>
  %v_view = buffer.view %v_global[%c0_offset] : buffer -> view<[%token_count]x[%qkv_stride]xf32>""",
             """  %q_view = buffer.view %q_global[%c0_offset] : buffer -> view<[%token_count]x[%qkv_stride]xf16>
  %k_view = buffer.view %k_global[%c0_offset] : buffer -> view<[%token_count]x[%qkv_stride]xf16>
  %v_view = buffer.view %v_global[%c0_offset] : buffer -> view<[%token_count]x[%qkv_stride]xf16>"""),
            ("""    %wide = vector.load %q_view[%query_bounded, %q_source] : view<[%token_count]x[%qkv_stride]xf32> -> vector<4xf32>
    %scaled = vector.mulf %wide, %scale_vector : vector<4xf32>
    %narrow = vector.fptrunc %scaled : vector<4xf32> to vector<4xf16>""",
             """    %half = vector.load %q_view[%query_bounded, %q_source] : view<[%token_count]x[%qkv_stride]xf16> -> vector<4xf16>
    %wide = vector.extf %half : vector<4xf16> to vector<4xf32>
    %scaled = vector.mulf %wide, %scale_vector : vector<4xf32>
    %narrow = vector.fptrunc %scaled : vector<4xf32> to vector<4xf16>"""),
            ("""        %wide = vector.load %k_view[%key_bounded, %q_source] : view<[%token_count]x[%qkv_stride]xf32> -> vector<4xf32>
        %narrow = vector.fptrunc %wide : vector<4xf32> to vector<4xf16>
        scf.yield %narrow : vector<4xf16>""",
             """        %narrow = vector.load %k_view[%key_bounded, %q_source] : view<[%token_count]x[%qkv_stride]xf16> -> vector<4xf16>
        scf.yield %narrow : vector<4xf16>"""),
            ("""        %wide = vector.load %v_view[%key_bounded, %q_source] : view<[%token_count]x[%qkv_stride]xf32> -> vector<4xf32>
        %narrow = vector.fptrunc %wide : vector<4xf32> to vector<4xf16>
        scf.yield %narrow : vector<4xf16>""",
             """        %narrow = vector.load %v_view[%key_bounded, %q_source] : view<[%token_count]x[%qkv_stride]xf16> -> vector<4xf16>
        scf.yield %narrow : vector<4xf16>"""),
        ],
        "// VARIANT: q/k/v arrive f16, so staging them is a straight copy.",
    ),
}


# Symbol suffix each variant appends, kept explicit so a rename cannot silently
# produce a kernel whose export name no longer matches the build script.
SUFFIX = {
    "layernorm_f32.loom": "_f16",
    "swiglu_f32.loom": "_f16",
    "flash_attention_f16_wmma.loom": "_cf16",
    "rope_2d_f32.loom": "_f16",
    "flash_attention_f16_wmma_cf16.loom": "_af16",
}


def main() -> None:
    for source, (target, symbol, namespace, edits, note) in EDITS.items():
        text = (KERNELS / source).read_text()
        suffix = SUFFIX[source]
        for old, new in edits:
            assert old in text, f"{source}: anchor moved\n{old[:80]}"
            text = text.replace(old, new)
        text = text.replace(symbol, symbol + suffix).replace(namespace, namespace + suffix)
        text = (note + f"\n// GENERATED by tools/gen_f16_variants.py from {source}.\n" + text)
        (KERNELS / target).write_text(text)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
