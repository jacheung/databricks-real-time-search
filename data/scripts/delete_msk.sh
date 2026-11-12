#!/usr/bin/env bash
# Delete everything data/scripts/create_msk.sh built: UC service credential, IAM role, MSK cluster and configuration,
# security group, subnets, internet gateway and VPC. Same variables as create_msk.sh (OWNER/REMOVE_AFTER not needed):
#
#   NAME=my-rts-msk AWS_PROFILE=<aws profile> DATABRICKS_PROFILE=<cli profile> data/scripts/delete_msk.sh
#
# DATABRICKS_PROFILE may list several profiles (space-separated): every workspace create_msk.sh was run for.
#
# Shared-config objects on top of it (the UC Kafka connection, the Feature Engineering Stream) are not touched:
# run teardown_job first if they should go too.
set -euo pipefail
: "${NAME:?}" "${AWS_PROFILE:?}" "${DATABRICKS_PROFILE:?}"
export AWS_PROFILE
REGION=${REGION:-us-west-2}
aws() { command aws --region "$REGION" --output text "$@"; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
none() { [ -z "$1" ] || [ "$1" = "None" ]; }

for p in $DATABRICKS_PROFILE; do
  databricks credentials delete-credential "$NAME" -p "$p" --force 2>/dev/null && log "deleted credential $NAME ($p)" || true
done
if command aws iam get-role --role-name "$NAME-access" >/dev/null 2>&1; then
  command aws iam delete-role-policy --role-name "$NAME-access" --policy-name msk-access || true
  command aws iam delete-role --role-name "$NAME-access" && log "deleted role $NAME-access"
fi

CLUSTER_ARN=$(aws kafka list-clusters-v2 --cluster-name-filter "$NAME" --query "ClusterInfoList[?ClusterName=='$NAME'].ClusterArn | [0]")
if ! none "$CLUSTER_ARN"; then
  [ "$(aws kafka describe-cluster-v2 --cluster-arn "$CLUSTER_ARN" --query ClusterInfo.State)" = "DELETING" ] \
    || aws kafka delete-cluster --cluster-arn "$CLUSTER_ARN" >/dev/null
  while aws kafka describe-cluster-v2 --cluster-arn "$CLUSTER_ARN" >/dev/null 2>&1; do log "cluster deleting"; sleep 60; done
  log "deleted cluster $NAME"
fi
CONFIG_ARN=$(aws kafka list-configurations --query "Configurations[?Name=='$NAME-config'].Arn | [0]")
none "$CONFIG_ARN" || { aws kafka delete-configuration --arn "$CONFIG_ARN" >/dev/null; log "deleted configuration"; }

VPC=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=$NAME-vpc" --query 'Vpcs[0].VpcId')
if ! none "$VPC"; then
  # MSK's broker ENIs can linger a few minutes after the cluster is gone; retry until the SG is free.
  SG=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC" "Name=group-name,Values=$NAME-sg" --query 'SecurityGroups[0].GroupId')
  if ! none "$SG"; then
    until aws ec2 delete-security-group --group-id "$SG" 2>/dev/null; do log "security group still in use"; sleep 30; done
  fi
  for sn in $(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC" --query 'Subnets[].SubnetId'); do
    aws ec2 delete-subnet --subnet-id "$sn"
  done
  for igw in $(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$VPC" --query 'InternetGateways[].InternetGatewayId'); do
    aws ec2 detach-internet-gateway --internet-gateway-id "$igw" --vpc-id "$VPC"
    aws ec2 delete-internet-gateway --internet-gateway-id "$igw"
  done
  aws ec2 delete-vpc --vpc-id "$VPC" && log "deleted vpc $VPC"
fi
