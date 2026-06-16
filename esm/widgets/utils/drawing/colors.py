def hex_to_rgba_tuple(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return r, g, b, alpha


def rgba_tuple_to_rgba_html_string(
    rgba_tuple: tuple[int | float, int | float, int | float, float],
) -> str:
    return f"rgba({rgba_tuple[0]},{rgba_tuple[1]},{rgba_tuple[2]},{rgba_tuple[3]})"


def rgba_tuple_to_hex(
    rgba: tuple[int | float, int | float, int | float]
    | tuple[int | float, int | float, int | float, int | float],
) -> str:
    def float_to_int(f):
        return int(f * 255)

    if all([isinstance(c, float) for c in rgba]):
        r = float_to_int(rgba[0])
        g = float_to_int(rgba[1])
        b = float_to_int(rgba[2])
        if len(rgba) > 3:
            rgba = (r, g, b, rgba[3])  # ty:ignore[index-out-of-bounds]
        else:
            rgba = (r, g, b)

    if len(rgba) == 4:
        rgba_ = (*rgba[:3], float_to_int(rgba[3]))
        return "#%02x%02x%02x%02x" % rgba_
    else:
        return "#%02x%02x%02x" % rgba
