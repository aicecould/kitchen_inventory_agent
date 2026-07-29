$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $root ".run\fasttext-src"
$build = Join-Path $root ".run\fasttext-build"
$cmakeFallback = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

$cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCommand) {
    $cmake = $cmakeCommand.Source
} elseif (Test-Path -LiteralPath $cmakeFallback) {
    $cmake = $cmakeFallback
} else {
    throw "CMake was not found. Install Visual Studio C++ Build Tools with CMake support."
}

if (-not (Test-Path -LiteralPath (Join-Path $source ".git"))) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $source) -Force | Out-Null
    git clone --depth 1 --branch v0.9.2 https://github.com/facebookresearch/fastText.git $source
}

$cmakeLists = Join-Path $source "CMakeLists.txt"
$content = Get-Content -LiteralPath $cmakeLists -Raw
$oldLink = "target_link_libraries(fasttext-bin pthread fasttext-static)"
if ($content.Contains($oldLink)) {
    $newLink = @"
if(WIN32)
  target_link_libraries(fasttext-bin fasttext-static)
else()
  target_link_libraries(fasttext-bin pthread fasttext-static)
endif()
"@
    Set-Content -LiteralPath $cmakeLists -Value ($content.Replace($oldLink, $newLink)) -Encoding UTF8
}

& $cmake -S $source -B $build -A x64
& $cmake --build $build --config Release --target fasttext-bin

$binary = Join-Path $build "Release\fasttext.exe"
if (-not (Test-Path -LiteralPath $binary)) {
    throw "FastText build completed without producing $binary"
}
Write-Output $binary
