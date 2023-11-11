using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using System.Windows.Forms;

namespace Lab4
{
    public partial class Form1 : Form
    {
        private List<Thread> threads;
        private List<bool> isRunning;

        public Form1()
        {
            InitializeComponent();
            threads = new List<Thread>();
            isRunning = new List<bool>
            {
                true,
                true,
                true,
                true
            };
            buttonStart1.Enabled = false;
            buttonStart2.Enabled = false;
            buttonStart3.Enabled = false;
            buttonStart4.Enabled = false;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            threads.Add(new Thread(Section_1Draw));
            threads.Add(new Thread(Section_2Draw));
            threads.Add(new Thread(Section_3Draw));
            threads.Add(new Thread(Section_4Draw));
            threads.ForEach(p => p.Start());
        }

        public void Section_1Draw()
        {
            Graphics g = panel1.CreateGraphics();
            var size = 50;
            var maxSize = 150;
            var minSize = 10;
            var increment = 5;
            var growing = true;

            while (true)
            {
                g.Clear(panel1.BackColor);

                var x = (panel1.Width - size) / 2;
                var y = (panel1.Height - size) / 2;

                g.FillRectangle(Brushes.Black, x, y, size, size);

                if (growing)
                {
                    size += increment;
                    if (size >= maxSize)
                        growing = false;
                }
                else
                {
                    size -= increment;
                    if (size <= minSize)
                        growing = true;
                }

                Thread.Sleep(300);
            }
        }

        private void Start_1(object sender, EventArgs e)
        {
            if (!isRunning[0])
            {
                threads[0].Resume();
                isRunning[0] = true;
            }
            buttonStop1.Enabled = true;
            buttonStart1.Enabled = false;
        }

        private void Stop_1(object sender, EventArgs e)
        {
            if (isRunning[0])
            {
                threads[0].Suspend();
                isRunning[0] = false;
            }
            buttonStart1.Enabled = true;
            buttonStop1.Enabled = false;

        }

        public bool IsInRectangle(ref double X, ref double Y, ref double dX, ref double dY, int radius, int width, int height)
        {
            if (X <= radius && dX < 0)
            {
                dX = -dX;
                X += dX;
                Y += dY;
                return false;
            }
            else if (X >= width - radius && dX > 0)
            {
                dX = -dX;
                X += dX;
                Y += dY;
                return false;
            }
            else if (Y <= radius && dY < 0)
            {
                dY = -dY;
                X += dX;
                Y += dY;
                return false;
            }
            else if (Y >= height - radius && dY > 0)
            {
                dY = -dY;
                X += dX;
                Y += dY;
                return false;
            }
            else
            {
                X += dX;
                Y += dY;
                return true;
            }
        }

        public void Section_2Draw()
        {
            Graphics g = panel2.CreateGraphics();
            Pen pen = new Pen(Color.Black, 2F);

            float x1 = 0, y1 = 0, x = 0;
            float y2;
            float yEx = 150;
            float eF = 25;

            g.Clear(panel2.BackColor);

            while (true)
            {
                y2 = (float)Math.Sin(x);
                g.DrawLine(pen, x1 * eF, y1 * eF + yEx, x * eF, y2 * eF + yEx);

                x1 = x;
                y1 = y2;
                x += 0.2f;

                if (x * eF >= 350)
                {
                    x = x1 = y1 = 0;
                    g.Clear(panel2.BackColor);
                }
                Thread.Sleep(10);
            }

            
        }

        private void Start_2(object sender, EventArgs e)
        {
            if (!isRunning[1])
            {
                threads[1].Resume();
                isRunning[1] = true;
            }
            buttonStop2.Enabled = true;
            buttonStart2.Enabled = false;
        }

        private void Stop_2(object sender, EventArgs e)
        {
            if (isRunning[1])
            {
                threads[1].Suspend();
                isRunning[1] = false;
            }
            buttonStart2.Enabled = true;
            buttonStop2.Enabled = false;
        }

        public void Section_3Draw()
        {
            Bitmap bitmap = new Bitmap(panel3.Width, panel3.Height);
            Graphics gImg = Graphics.FromImage(bitmap);
            Graphics graphics = panel3.CreateGraphics();
            PointF img = new PointF(0, 0);
            Random rand = new Random();

            double X, Y, dX, dY;
            int radius = 25;
            X = rand.Next(radius, panel3.Width - radius);
            Y = rand.Next(radius, panel3.Height - radius);
            dX = (rand.Next(1, 5)) * 0.1;
            dY = (rand.Next(1, 5)) * 0.1;

            while (true)
            {
                gImg.Clear(panel3.BackColor);
                gImg.FillEllipse(new SolidBrush(Color.FromArgb(0, 0, 0)), (float)X - radius, (float)Y - radius, radius * 2, radius * 2);
                IsInRectangle(ref X, ref Y, ref dX, ref dY, radius, panel3.Width, panel3.Height);
                graphics.DrawImage(bitmap, img);
            }
        }

        private void Start_3(object sender, EventArgs e)
        {
            if (!isRunning[2])
            {
                threads[2].Resume();
                isRunning[2] = true;
            }
            buttonStop3.Enabled = true;
            buttonStart3.Enabled = false;
        }

        private void Stop_3(object sender, EventArgs e)
        {
            if (isRunning[2])
            {
                threads[2].Suspend();
                isRunning[2] = false;
            }
            buttonStart3.Enabled = true;
            buttonStop3.Enabled = false;
        }

        public void Section_4Draw()
        {
            Graphics g = panel4.CreateGraphics();
            int[] arrayToSort = new int[20];
            Random random = new Random();

            while (true)
            {
                for (int i = 0; i < arrayToSort.Length; i++)
                {
                    arrayToSort[i] = random.Next(10, 250);
                }

                bool sorted = false;
                while (!sorted)
                {
                    g.Clear(panel4.BackColor);

                    int barWidth = panel4.Width / arrayToSort.Length;
                    for (int i = 0; i < arrayToSort.Length; i++)
                    {
                        int barHeight = arrayToSort[i];
                        Rectangle rect = new Rectangle(i * barWidth, panel4.Height - barHeight, barWidth, barHeight);
                        g.FillRectangle(Brushes.Black, rect);
                    }

                    sorted = true;
                    for (int i = 0; i < arrayToSort.Length - 1; i++)
                    {
                        for (int j = 0; j < arrayToSort.Length - 1 - i; j++)
                        {
                            if (arrayToSort[j] > arrayToSort[j + 1])
                            {
                                int temp = arrayToSort[j];
                                arrayToSort[j] = arrayToSort[j + 1];
                                arrayToSort[j + 1] = temp;

                                g.Clear(panel4.BackColor);
                                for (int k = 0; k < arrayToSort.Length; k++)
                                {
                                    int barHeight = arrayToSort[k];
                                    Rectangle rect = new Rectangle(k * barWidth, panel4.Height - barHeight, barWidth, barHeight);
                                    g.FillRectangle(Brushes.Black, rect);
                                }

                                sorted = false;
                                Thread.Sleep(100);
                            }
                        }
                    }

                    Thread.Sleep(500);
                }
            }
        }

        private void Start_4(object sender, EventArgs e)
        {
            if (!isRunning[3])
            {
                threads[3].Resume();
                isRunning[3] = true;
            }
            buttonStop4.Enabled = true;
            buttonStart4.Enabled = false;
        }

        private void Stop_4(object sender, EventArgs e)
        {
            if (isRunning[3])
            {
                threads[3].Suspend();
                isRunning[3] = false;
            }
            buttonStart4.Enabled = true;
            buttonStop4.Enabled = false;
        }

        private void MainForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            for(int i = 0; i < 4; i++)
            {
                if (!isRunning[i])
                {
                    threads[i].Resume();
                    isRunning[i] = true;
                }
            }
            threads.ForEach(p => p.Abort());
        }
    }
}
