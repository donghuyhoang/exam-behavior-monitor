using System.Net.Http;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;

namespace CameraMonitorApp
{
    public partial class MainWindow : Window
    {
        private const string StatusUrl = "http://localhost:8000/status";
        private const double PhoneThreshold = 0.8;   // 80% (conf từ Python nằm trong khoảng 0..1)

        private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(2) };
        private readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(500) };
        private bool _polling;

        private static readonly Brush NormalBrush = new SolidColorBrush(Color.FromRgb(0x4F, 0xD1, 0xC5));
        private static readonly Brush AlertBrush = new SolidColorBrush(Color.FromRgb(0xF5, 0x6565 >> 8, 0x65));

        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;
        }

        private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            await webView.EnsureCoreWebView2Async();
            webView.NavigateToString(@"
        <html><body style='margin:0'>
            <img src='http://localhost:8000/video' style='width:100%;height:100vh;object-fit:contain'>
        </body></html>");

            _timer.Tick += async (_, _) => await PollStatusAsync();
            _timer.Start();
        }

        private async Task PollStatusAsync()
        {
            if (_polling) return;   // tránh gọi chồng khi request chưa xong
            _polling = true;
            try
            {
                string json = await _http.GetStringAsync(StatusUrl);
                using var doc = JsonDocument.Parse(json);
                bool phone = doc.RootElement.GetProperty("phone").GetBoolean();
                double conf = doc.RootElement.GetProperty("conf").GetDouble();

                if (phone && conf > PhoneThreshold)
                    SetStatus($"PHÁT HIỆN SỬ DỤNG ĐIỆN THOẠI ({conf:P0})", AlertBrush);
                else
                    SetStatus("Trạng thái: Bình thường", NormalBrush);
            }
            catch
            {
                // Python server chưa chạy hoặc lỗi mạng => giữ nguyên, thử lại lần sau
            }
            finally
            {
                _polling = false;
            }
        }

        private void SetStatus(string text, Brush color)
        {
            statusText.Text = text;
            statusDot.Fill = color;
            statusText.Foreground = color == AlertBrush ? AlertBrush : Brushes.Gainsboro;
        }
    }
}