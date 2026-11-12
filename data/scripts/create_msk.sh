#!/usr/bin/env bash
# Create a public Amazon MSK cluster (IAM auth) for the session-event Kafka topic, plus the UC service credential
# Databricks serverless uses to reach it. Idempotent: rerun to resume after a failure. ~1 h end to end.
#
#   NAME=my-rts-msk OWNER=me@example.com REMOVE_AFTER=2026-12-11 AWS_PROFILE=<aws profile> DATABRICKS_PROFILE=<cli profile> \
#     data/scripts/create_msk.sh
#
# Builds, in the AWS account of AWS_PROFILE (region REGION, default us-west-2), all tagged Owner/Project/RemoveAfter:
#   VPC <NAME>-vpc (VPC_CIDR) + internet gateway + 3 public subnets, one per AZ
#   security group <NAME>-sg: 9098 from itself; 9198 (public SASL/IAM) only from Databricks serverless egress for REGION
#   MSK configuration <NAME>-config: allow.everyone.if.no.acl.found=false (required for public access)
#   MSK cluster <NAME>: provisioned, 3 x kafka.m7g.large, Kafka 3.9.x, 100 GiB/broker, IAM auth, TLS; then public access
#   IAM role <NAME>-access: kafka-cluster:* on this cluster's topics and groups, trusted by Unity Catalog
#   UC service credential <NAME> in the workspace of DATABRICKS_PROFILE
# Prints the kafka: block for shared/configs/<target>.yml.
# Another workspace (same region): rerun with its DATABRICKS_PROFILE; the role then trusts both credentials. Give
# each workspace its own kafka.topic, or their Streams ingest each other's events.
#
# Bills ~$15/day while it exists; MSK can't be stopped. data/scripts/delete_msk.sh removes everything.
# Egress ranges: https://www.databricks.com/networking/v1/ip-ranges.json (type outbound). Clients are the app, the
# Feature Engineering Stream, the session pipeline and notebooks, all serverless, so all ranges are allowed.
set -euo pipefail
: "${NAME:?}" "${OWNER:?}" "${REMOVE_AFTER:?}" "${AWS_PROFILE:?}" "${DATABRICKS_PROFILE:?}"
export AWS_PROFILE
REGION=${REGION:-us-west-2}
VPC_CIDR=${VPC_CIDR:-10.71.0.0/16}     # pick one no other VPC in the account uses
PREFIX=${VPC_CIDR%.*.*}                # 10.71
TAGS="Key=Owner,Value=$OWNER Key=Project,Value=real-time-search Key=RemoveAfter,Value=$REMOVE_AFTER"
aws() { command aws --region "$REGION" --output text "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
tag_spec() { echo "ResourceType=$1,Tags=[{Key=Name,Value=$2},{Key=Owner,Value=$OWNER},{Key=Project,Value=real-time-search},{Key=RemoveAfter,Value=$REMOVE_AFTER}]"; }
none() { [ -z "$1" ] || [ "$1" = "None" ]; }
ACCOUNT=$(aws sts get-caller-identity --query Account)

# --- Network
VPC=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=$NAME-vpc" --query 'Vpcs[0].VpcId')
if none "$VPC"; then
  VPC=$(aws ec2 create-vpc --cidr-block "$VPC_CIDR" --tag-specifications "$(tag_spec vpc "$NAME-vpc")" --query Vpc.VpcId)
  aws ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-hostnames
fi
log "vpc $VPC"

IGW=$(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$VPC" --query 'InternetGateways[0].InternetGatewayId')
if none "$IGW"; then
  IGW=$(aws ec2 create-internet-gateway --tag-specifications "$(tag_spec internet-gateway "$NAME-igw")" --query InternetGateway.InternetGatewayId)
  aws ec2 attach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC"
fi
RT=$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$VPC" "Name=association.main,Values=true" --query 'RouteTables[0].RouteTableId')
aws ec2 create-route --route-table-id "$RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW" >/dev/null 2>&1 \
  || aws ec2 replace-route --route-table-id "$RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW"
log "igw $IGW, route table $RT"

SUBNETS=()
i=0
for az in a b c; do
  sn=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" "Name=availability-zone,Values=$REGION$az" --query 'Subnets[0].SubnetId')
  if none "$sn"; then
    sn=$(aws ec2 create-subnet --vpc-id "$VPC" --cidr-block "$PREFIX.$i.0/24" --availability-zone "$REGION$az" \
      --tag-specifications "$(tag_spec subnet "$NAME-public-$az")" --query Subnet.SubnetId)
    aws ec2 modify-subnet-attribute --subnet-id "$sn" --map-public-ip-on-launch
  fi
  SUBNETS+=("$sn"); i=$((i + 1))
done
log "subnets ${SUBNETS[*]}"

SG=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC" "Name=group-name,Values=$NAME-sg" --query 'SecurityGroups[0].GroupId')
if none "$SG"; then
  SG=$(aws ec2 create-security-group --vpc-id "$VPC" --group-name "$NAME-sg" --description "MSK $NAME: Databricks serverless egress" \
    --tag-specifications "$(tag_spec security-group "$NAME-sg")" --query GroupId)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 9098 --source-group "$SG" >/dev/null
  CIDRS=$(curl -sSf https://www.databricks.com/networking/v1/ip-ranges.json | python3 -I -c "
import json, sys
d = json.load(sys.stdin)
print(' '.join(c for p in d['prefixes'] if p['platform'] == 'aws' and p['region'] == '$REGION' and p['type'] == 'outbound'
               for c in p['ipv4Prefixes']))")
  [ -n "$CIDRS" ] || { echo "no Databricks outbound ranges for $REGION"; exit 1; }
  PERMS=$(python3 -I -c "import json,sys; print(json.dumps([{'IpProtocol': 'tcp', 'FromPort': 9198, 'ToPort': 9198,
    'IpRanges': [{'CidrIp': c, 'Description': 'Databricks serverless egress'} for c in sys.argv[1:]]}]))" $CIDRS)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --ip-permissions "$PERMS" >/dev/null
  log "9198 open to: $CIDRS"
fi
log "security group $SG"

# --- MSK configuration and cluster
CONFIG_ARN=$(aws kafka list-configurations --query "Configurations[?Name=='$NAME-config'].Arn | [0]")
if none "$CONFIG_ARN"; then
  props=$(mktemp); echo "allow.everyone.if.no.acl.found=false" > "$props"
  CONFIG_ARN=$(aws kafka create-configuration --name "$NAME-config" --kafka-versions 3.9.x --server-properties "fileb://$props" --query Arn)
  rm -f "$props"
fi
log "configuration $CONFIG_ARN"

CLUSTER_ARN=$(aws kafka list-clusters-v2 --cluster-name-filter "$NAME" --query "ClusterInfoList[?ClusterName=='$NAME'].ClusterArn | [0]")
if none "$CLUSTER_ARN"; then
  PROVISIONED=$(python3 -I -c "import json,sys; print(json.dumps({
    'BrokerNodeGroupInfo': {'InstanceType': 'kafka.m7g.large', 'ClientSubnets': sys.argv[3:], 'SecurityGroups': [sys.argv[1]],
                            'StorageInfo': {'EbsStorageInfo': {'VolumeSize': 100}}},
    'KafkaVersion': '3.9.x', 'NumberOfBrokerNodes': 3,
    'ClientAuthentication': {'Sasl': {'Iam': {'Enabled': True}}, 'Unauthenticated': {'Enabled': False}},
    'EncryptionInfo': {'EncryptionInTransit': {'ClientBroker': 'TLS', 'InCluster': True}},
    'ConfigurationInfo': {'Arn': sys.argv[2], 'Revision': 1}}))" "$SG" "$CONFIG_ARN" "${SUBNETS[@]}")
  CLUSTER_ARN=$(aws kafka create-cluster-v2 --cluster-name "$NAME" --provisioned "$PROVISIONED" \
    --tags "Owner=$OWNER,Project=real-time-search,RemoveAfter=$REMOVE_AFTER" --query ClusterArn)
fi
log "cluster $CLUSTER_ARN"

wait_active() {
  local state
  while :; do
    # A transient network error shouldn't end an hour-long wait: retry the status call.
    state=$(aws kafka describe-cluster-v2 --cluster-arn "$CLUSTER_ARN" --query ClusterInfo.State) \
      || { log "status check failed, retrying"; sleep 30; continue; }
    [ "$state" = "ACTIVE" ] && return
    [ "$state" = "FAILED" ] && { echo "cluster FAILED"; exit 1; }
    log "cluster $state"; sleep 60
  done
}
wait_active
PUBLIC=$(aws kafka describe-cluster-v2 --cluster-arn "$CLUSTER_ARN" --query ClusterInfo.Provisioned.BrokerNodeGroupInfo.ConnectivityInfo.PublicAccess.Type)
if [ "$PUBLIC" != "SERVICE_PROVIDED_EIPS" ]; then
  VERSION=$(aws kafka describe-cluster-v2 --cluster-arn "$CLUSTER_ARN" --query ClusterInfo.CurrentVersion)
  aws kafka update-connectivity --cluster-arn "$CLUSTER_ARN" --current-version "$VERSION" \
    --connectivity-info '{"PublicAccess":{"Type":"SERVICE_PROVIDED_EIPS"}}' >/dev/null
  log "public access requested"; sleep 60
  wait_active
fi
BOOTSTRAP=$(aws kafka get-bootstrap-brokers --cluster-arn "$CLUSTER_ARN" --query BootstrapBrokerStringPublicSaslIam)
log "public bootstrap $BOOTSTRAP"

# --- IAM role, trusted by Unity Catalog
ROLE=$NAME-access
ROLE_ARN=arn:aws:iam::$ACCOUNT:role/$ROLE
CLUSTER_PATH=${CLUSTER_ARN#*:cluster/}   # <name>/<uuid>
if ! command aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  # Placeholder trust; replaced below once the UC credential exists and returns its external ID.
  command aws iam create-role --role-name "$ROLE" --tags $TAGS --output text --query Role.Arn \
    --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"arn:aws:iam::$ACCOUNT:root\"},\"Action\":\"sts:AssumeRole\"}]}"
fi
command aws iam put-role-policy --role-name "$ROLE" --policy-name msk-access --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"kafka-cluster:*\",\"Resource\":[
  \"$CLUSTER_ARN\",
  \"arn:aws:kafka:$REGION:$ACCOUNT:topic/$CLUSTER_PATH/*\",
  \"arn:aws:kafka:$REGION:$ACCOUNT:group/$CLUSTER_PATH/*\"]},
  {\"Effect\":\"Allow\",\"Action\":\"sts:AssumeRole\",\"Resource\":\"$ROLE_ARN\"}]}"   # UC checks the role can assume itself
log "role $ROLE_ARN"

# --- UC service credential, then trust UC with its external ID (the role also trusts itself, as UC requires).
# Merged into the existing trust, so each workspace that reruns this keeps the others' access.
if ! databricks credentials get-credential "$NAME" -p "$DATABRICKS_PROFILE" >/dev/null 2>&1; then
  databricks credentials create-credential -p "$DATABRICKS_PROFILE" -o json --json "{\"name\":\"$NAME\",\"purpose\":\"SERVICE\",
    \"aws_iam_role\":{\"role_arn\":\"$ROLE_ARN\"},\"skip_validation\":true,\"comment\":\"MSK $NAME (session events)\"}" >/dev/null
fi
read -r UC_ARN EXTERNAL_ID < <(databricks credentials get-credential "$NAME" -p "$DATABRICKS_PROFILE" -o json \
  | python3 -I -c "import json,sys; r=json.load(sys.stdin)['aws_iam_role']; print(r['unity_catalog_iam_arn'], r['external_id'])")
TRUST=$(command aws iam get-role --role-name "$ROLE" --query Role.AssumeRolePolicyDocument --output json | python3 -I -c "
import json, sys
doc, (uc, role, ext) = json.load(sys.stdin), sys.argv[1:]
principals, ids = {uc, role}, {ext}
for s in doc['Statement']:
    aws = s.get('Principal', {}).get('AWS', [])
    principals.update(p for p in ([aws] if isinstance(aws, str) else aws) if not p.endswith(':root'))  # drop the placeholder
    v = s.get('Condition', {}).get('StringEquals', {}).get('sts:ExternalId', [])
    ids.update([v] if isinstance(v, str) else v)
print(json.dumps({'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'AWS': sorted(principals)},
    'Action': 'sts:AssumeRole', 'Condition': {'StringEquals': {'sts:ExternalId': sorted(ids)}}}]}))" "$UC_ARN" "$ROLE_ARN" "$EXTERNAL_ID")
command aws iam update-assume-role-policy --role-name "$ROLE" --policy-document "$TRUST"
log "credential $NAME trusted by $UC_ARN (external IDs: $(python3 -I -c "import json,sys; print(json.loads(sys.argv[1])['Statement'][0]['Condition']['StringEquals']['sts:ExternalId'])" "$TRUST"))"

cat <<EOF

kafka:                             # for shared/configs/<target>.yml
  service_credential: $NAME
  bootstrap_servers: "$BOOTSTRAP"
EOF
