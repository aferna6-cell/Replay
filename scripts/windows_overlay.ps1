# HDT-style native Windows overlay for the WSL coach.
#
# The coach (running in WSL) writes the panel text to a file and launches
# this script with powershell.exe. This window is a real Windows top-level
# WPF window — Topmost like HDT's own overlay — so it floats over a
# windowed/borderless Hearthstone and clicking the game does NOT hide it.
#
# Usage:  powershell.exe -NoProfile -ExecutionPolicy Bypass `
#           -File windows_overlay.ps1 -PanelFile <path readable from Windows>
#
# Drag anywhere on the panel to move it. Close it (right-click -> Close via
# Alt+F4 while focused) to stop; the coach keeps running in the terminal.
param(
    [Parameter(Mandatory = $true)][string]$PanelFile
)

Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName WindowsBase

$win = New-Object Windows.Window
$win.Title = "HSBG Coach"
$win.WindowStyle = "None"
$win.ResizeMode = "NoResize"
$win.AllowsTransparency = $true
$win.Background = New-Object Windows.Media.SolidColorBrush(
    [Windows.Media.Color]::FromArgb(225, 18, 18, 24))
$win.Topmost = $true
$win.ShowInTaskbar = $false
$win.ShowActivated = $false          # never steal focus from the game
$win.SizeToContent = "WidthAndHeight"
$win.MaxWidth = 620
$win.MaxHeight = 700

$text = New-Object Windows.Controls.TextBlock
$text.FontFamily = New-Object Windows.Media.FontFamily("Consolas")
$text.FontSize = 13
$text.Foreground = [Windows.Media.Brushes]::WhiteSmoke
$text.Margin = New-Object Windows.Thickness(12, 10, 12, 10)
$text.TextWrapping = "Wrap"
$text.Text = "HSBG Coach - waiting for the game..."
$win.Content = $text

# Drag anywhere to reposition (like HDT's panels).
$win.Add_MouseLeftButtonDown({ try { $win.DragMove() } catch {} })

# Start pinned to the top-right of the work area.
$wa = [Windows.SystemParameters]::WorkArea
$win.Left = $wa.Right - 640
$win.Top = $wa.Top + 8

$script:lastText = ""
$script:tick = 0
$timer = New-Object Windows.Threading.DispatcherTimer
$timer.Interval = [TimeSpan]::FromMilliseconds(250)
$timer.Add_Tick({
    try {
        if (Test-Path -LiteralPath $PanelFile) {
            $t = [IO.File]::ReadAllText($PanelFile)
            if ($t -ne $script:lastText) {
                $script:lastText = $t
                $text.Text = $t
            }
        }
        # Re-assert Topmost occasionally so the game can't cover the panel
        # after focus changes (same trick HDT uses) — not every tick, to
        # avoid z-order churn/flicker.
        $script:tick++
        if ($script:tick % 8 -eq 0) {
            $win.Topmost = $false
            $win.Topmost = $true
        }
    } catch {}
})
$timer.Start()

[void]$win.ShowDialog()
