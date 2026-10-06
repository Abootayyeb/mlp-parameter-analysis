"""
محاسبه‌گر جامع پارامترهای آموزش‌پذیر در شبکه‌های پرسپترون چندلایه (MLP)
مناسب برای بررسی تکالیف و مسائل درسی رایانش عصبی و یادگیری عمیق.
"""

from typing import List, Union


def analyze_mlp(
    layers: List[int],
    use_bias: Union[bool, List[bool]] = True,
    has_batchnorm: bool = False,
):
    """
    تحلیل کامل معماری شبکه MLP.

    layers: ابعاد لایه‌ها [ورودی, پنهان1, ..., خروجی]
    use_bias: وضعیت بایاس (یک مقدار بولی برای همه لایه‌ها یا لیستی از بولی‌ها)
    has_batchnorm: آیا بعد از لایه‌های پنهان BatchNorm1d وجود دارد؟
    """
    num_transitions = len(layers) - 1

    # یکسان‌سازی وضعیت بایاس برای تمام لایه‌ها
    if isinstance(use_bias, bool):
        biases_config = [use_bias] * num_transitions
    else:
        biases_config = use_bias

    total_weights = 0
    total_biases = 0
    total_bn_params = 0

    print("=" * 72)
    print(f"معماری شبکه: {' -> '.join(map(str, layers))}")
    print("=" * 72)
    print(
        f"{'لایه':<12} | {'ابعاد':<12} | {'وزن‌ها':<10} | {'بایاس':<8} | {'بچ‌نرم':<8} | {'مجموع':<8}"
    )
    print("-" * 72)

    for i in range(num_transitions):
        n_in = layers[i]
        n_out = layers[i + 1]
        b_flag = biases_config[i]

        w = n_in * n_out
        b = n_out if b_flag else 0
        bn = (2 * n_out) if (has_batchnorm and i < num_transitions - 1) else 0

        layer_total = w + b + bn
        total_weights += w
        total_biases += b
        total_bn_params += bn

        layer_name = f"لایه {i+1}"
        dim_str = f"{n_in} -> {n_out}"
        print(
            f"{layer_name:<12} | {dim_str:<12} | {w:<10} | {b:<8} | {bn:<8} | {layer_total:<8}"
        )

    grand_total = total_weights + total_biases + total_bn_params
    storage_kb = (grand_total * 4) / 1024
    adam_ram_kb = (grand_total * 16) / 1024

    print("=" * 72)
    print(f"مجموع وزن‌ها:                   {total_weights:,}")
    print(f"مجموع بایاس‌ها:                  {total_biases:,}")
    if has_batchnorm:
        print(f"پارامترهای BatchNorm:            {total_bn_params:,}")
    print(f"کل پارامترهای آموزش‌پذیر:        {grand_total:,}")
    print("-" * 72)
    print(f"حجم فایل مدل (FP32):             {storage_kb:.2f} KB ({grand_total * 4:,} Bytes)")
    print(f"حافظه مورد نیاز در آموزش (Adam):  ~{adam_ram_kb:.2f} KB ({grand_total * 16:,} Bytes)")
    print("=" * 72 + "\n")

    return grand_total


if __name__ == "__main__":
    # سناریو ۱: مثال اول جزوه (ورودی ۸، پنهان ۱۶، پنهان ۳۲، خروجی ۴)
    print("--- سناریو ۱: شبکه استاندارد ---")
    analyze_mlp(layers=[8, 16, 32, 4], use_bias=True)

    # سناریو ۲: شبکه با لایه‌های بدون بایاس یا ترکیبی
    # لایه اول بایاس دارد، لایه دوم بدون بایاس است
    print("--- سناریو ۲: شبکه با بایاس دلخواه ---")
    analyze_mlp(layers=[10, 20, 5], use_bias=[True, False])
