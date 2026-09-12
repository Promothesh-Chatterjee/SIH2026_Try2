
with open('c:/HACKATHONS/SIH2026_Try2/SMARTSCAN_EW_FRONTEND_APP/src/services/useOverviewTelemetry.js', 'r', encoding='utf-8') as f:
    code = f.read()

import re
code = re.sub(r'incidentPdwsRef\.current\.map\(\(p\) => \$\{p\.pulse_id \?\? p\.id\}-\)', 'incidentPdwsRef.current.map((p) => \\-\\)', code)
code = re.sub(r'const uid = \$\{p\.pulse_id \?\? p\.id\}-;', 'const uid = \\-\\;', code)

with open('c:/HACKATHONS/SIH2026_Try2/SMARTSCAN_EW_FRONTEND_APP/src/services/useOverviewTelemetry.js', 'w', encoding='utf-8') as f:
    f.write(code)

