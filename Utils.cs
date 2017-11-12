using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace SelfOWatch.Utils
{
    public class Fov
    {
        public Point resolution { get; set; }

        public Rectangle field_of_view { get; set; }

        public Point range_values { get; set; }

        public Point tolerance { get; set; }
    }

    public class Screen
    {
        [DllImport("gdi32.dll")]
        private static extern bool BitBlt(IntPtr dc_dest, int x_dest, int y_dest, int width, int height, IntPtr dc_src, int x_src, int y_src, int rop);

        public static Bitmap GetScreenCapture(Rectangle fov)
        {
            var screen_copy = new Bitmap(fov.Width, fov.Height, PixelFormat.Format24bppRgb);

            using (var g_dest = Graphics.FromImage(screen_copy))

            using (var g_src = Graphics.FromHwnd(IntPtr.Zero))
            {
                var src_dc = g_src.GetHdc();
                var dc = g_dest.GetHdc();
                var retval = BitBlt(dc, 0, 0, fov.Width, fov.Height, src_dc, fov.X, fov.Y, (int)CopyPixelOperation.SourceCopy);

                g_dest.ReleaseHdc();
                g_src.ReleaseHdc();
            }
            return screen_copy;
        }

        public static Point GetAbsoluteCoordinates(Point relative, Rectangle fov)
        {
            return new Point { X = relative.X + fov.X, Y = relative.Y + fov.Y };
        }
    }
}
