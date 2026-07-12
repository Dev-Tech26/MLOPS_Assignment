{{- define "heart-disease.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "heart-disease.fullname" -}}
{{- .Release.Name }}-{{ include "heart-disease.name" . }}
{{- end }}

{{- define "heart-disease.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{ include "heart-disease.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "heart-disease.selectorLabels" -}}
app.kubernetes.io/name: {{ include "heart-disease.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
