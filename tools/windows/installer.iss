; Inno Setup script for Workbench

[Setup]
AppId={{C7B2E3D4-5F6A-7B8C-9D0E-1F2A3B4C5D6E}
AppName=Workbench
AppVersion=0.1.0
AppPublisher=Bakhtiyor Sohibnazarov
DefaultDirName={autopf}\Workbench
DefaultGroupName=Workbench
AllowNoIcons=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename=Workbench_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\dist\Workbench\Workbench.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\Workbench\libs\*"; DestDir: "{app}\libs"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\core\*"; DestDir: "{app}\core"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Workbench"; Filename: "{app}\Workbench.exe"
Name: "{autodesktop}\Workbench"; Filename: "{app}\Workbench.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Workbench.exe"; Description: "{cm:LaunchProgram,Workbench}"; Flags: nowait postinstall skipifsilent
