/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the user API definitions
 * for the SMBus driver.
 *
 * @file smbus_driver.c
 *
 */
#include <string.h>
#include "smbus.h"
#include "i2c.h"
#include "smbus_internal.h"
#include "smbus_hardware_access.h"
#include "smbus_event.h"
#include "smbus_event_buffer.h"
#include "smbus_action.h"
#include "smbus_state.h"
#include "smbus_interrupt_handler.h"

#include "smbus_version.h"

static SMBUS_PROFILE_TYPE xSMBusProfile =
{
       .pFnReadTicks           = 0,
       .pvBaseAddress          = 0,
       .ulTransactionID        = 0,
       .ulInitialize           = 0,
       .xLogCircularBuffer     = { 0 },
       .xCircularBuffer        = { { 0 } },
       .xSMBusInstance         = { { 0 } },
       .ucInstanceInPlay       = SMBUS_INVALID_INSTANCE,
       .ucActiveTargetInstance = SMBUS_INVALID_INSTANCE,
       .xLogLevel              = 0,
       .ucUDIDMatch            = { 0 }
 };

static char* prvpcProtocol_SMBUS_PROTOCOL_QUICK_COMMAND_HI                     = "SMBUS_PROTOCOL_QUICK_COMMAND_HI";
static char* prvpcProtocol_SMBUS_PROTOCOL_QUICK_COMMAND_LO                     = "SMBUS_PROTOCOL_QUICK_COMMAND_LO";
static char* prvpcProtocol_SMBUS_PROTOCOL_SEND_BYTE                            = "SMBUS_PROTOCOL_SEND_BYTE";
static char* prvpcProtocol_SMBUS_PROTOCOL_RECEIVE_BYTE                         = "SMBUS_PROTOCOL_RECEIVE_BYTE";
static char* prvpcProtocol_SMBUS_PROTOCOL_WRITE_BYTE                           = "SMBUS_PROTOCOL_WRITE_BYTE";
static char* prvpcProtocol_SMBUS_PROTOCOL_WRITE_WORD                           = "SMBUS_PROTOCOL_WRITE_WORD";
static char* prvpcProtocol_SMBUS_PROTOCOL_READ_BYTE                            = "SMBUS_PROTOCOL_READ_BYTE";
static char* prvpcProtocol_SMBUS_PROTOCOL_READ_WORD                            = "SMBUS_PROTOCOL_READ_WORD";
static char* prvpcProtocol_SMBUS_PROTOCOL_PROCESS_CALL                         = "SMBUS_PROTOCOL_PROCESS_CALL";
static char* prvpcProtocol_SMBUS_PROTOCOL_BLOCK_WRITE                          = "SMBUS_PROTOCOL_BLOCK_WRITE";
static char* prvpcProtocol_SMBUS_PROTOCOL_BLOCK_READ                           = "SMBUS_PROTOCOL_BLOCK_READ";
static char* prvpcProtocol_SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL  = "SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL";
static char* prvpcProtocol_SMBUS_PROTOCOL_HOST_NOTIFY                          = "SMBUS_PROTOCOL_HOST_NOTIFY";
static char* prvpcProtocol_SMBUS_PROTOCOL_WRITE_32                             = "SMBUS_PROTOCOL_WRITE_32";
static char* prvpcProtocol_SMBUS_PROTOCOL_READ_32                              = "SMBUS_PROTOCOL_READ_32";
static char* prvpcProtocol_SMBUS_PROTOCOL_WRITE_64                             = "SMBUS_PROTOCOL_WRITE_64";
static char* prvpcProtocol_SMBUS_PROTOCOL_READ_64                              = "SMBUS_PROTOCOL_READ_64";
static char* prvpcProtocol_SMBUS_PROTOCOL_NONE                                 = "SMBUS_PROTOCOL_NONE";
static char* prvpcProtocol_SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP                   = "SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP";
static char* prvpcProtocol_SMBUS_ARP_PROTOCOL_RESET_DEVICE                     = "SMBUS_ARP_PROTOCOL_RESET_DEVICE";
static char* prvpcProtocol_SMBUS_ARP_PROTOCOL_GET_UDID                         = "SMBUS_ARP_PROTOCOL_GET_UDID";
static char* prvpcProtocol_SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS                   = "SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS";
static char* prvpcProtocol_SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED            = "SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED";
static char* prvpcProtocol_SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED                = "SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED";
static char* prvpcProtocol_UNKNOWN                                             = "SMBUS_PROTOCOL_UNKNOWN";

/* Driver Functions */

/*******************************************************************************
*
* @brief    Does a ceiling conversion on a floating point number and returns the
*           rounded up interger value
*
*******************************************************************************/
uint32_t ulSMBusCeil( float fNum )
{
    uint32_t ulNum = ( uint32_t )fNum;
    if ( fNum != ( float )ulNum )
    {
        ulNum++;
    }

    return ( ulNum );
}

/*******************************************************************************
*
* @brief    Converts a protocol enum value to a text string for logging
*
*******************************************************************************/
SMBus_Error_Type xSMBusFirewallCheck( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    SMBus_Error_Type    xError = SMBUS_SUCCESS;
    int                 i = 0;

    if( NULL != pxSMBusProfile )
    {
        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_INSTANCES; i++ )
        {
            if( ( SMBUS_FIREWALL1 != pxSMBusProfile->xSMBusInstance[i].ulFirewall1 ) ||
                ( SMBUS_FIREWALL2 != pxSMBusProfile->xSMBusInstance[i].ulFirewall2 ) ||
                ( SMBUS_FIREWALL3 != pxSMBusProfile->xSMBusInstance[i].ulFirewall3 ) )
            {
                xError = SMBUS_ERROR;
                break;
            }
        }
    }
    else
    {
        xError = SMBUS_ERROR;
    }

    return xError;
}

/*******************************************************************************
*
* @brief    Will walk through all active instances, check if any events have been
*           raised against that instance and call into the state machine for that
*           instance with each event found
*
*******************************************************************************/
void vSMBusEventQueueHandle( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucAnyEvent;
    int i;
    uint32_t ulRead_Position;

    if( NULL != pxSMBusProfile )
    {
        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_INSTANCES; i++ )
        {
            if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
            {
                while( ( ucEventBufferTryRead( &( pxSMBusProfile->xSMBusInstance[i].xEventSourceCircularBuffer ),
                                            &ucAnyEvent, &ulRead_Position ) ) )
                {
                    vSMBusFSM( &( pxSMBusProfile->xSMBusInstance[i] ), ucAnyEvent );
                }
            }
        }
    }
}

/*******************************************************************************
*
* @brief    Retrieves the SMBus driver version
*
*****************************************************************************/
SMBus_Error_Type xSMBusGetVersion( struct SMBUS_PROFILE_TYPE* pxSMBusProfile, SMBUS_VERSION_TYPE* pxSMBusVersion )
{
#ifdef GIT_TAG
    SMBus_Error_Type xError = SMBUS_ERROR;

    if( NULL != pxSMBusVersion )
    {
        pxSMBusVersion->ucSwVerMajor = GIT_TAG_VER_MAJOR;
        pxSMBusVersion->ucSwVerMinor = GIT_TAG_VER_MINOR;
        pxSMBusVersion->ucSwVerPatch = GIT_TAG_VER_PATCH;
        pxSMBusVersion->ucSwDevBuild = GIT_TAG_VER_DEV_COMMITS;
        pxSMBusVersion->ucSwTestBuild = ( 0 == GIT_STATUS ) ? ( 0 ):( 1 );

        if( NULL != pxSMBusProfile )
        {
            if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
            {
                vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                SMBUS_ERROR, __LINE__ );
                pxSMBusVersion->usIpVerMajor = 0;
                pxSMBusVersion->usIpVerMinor = 0;
            }
            else
            {
                uint32_t ulIpVersion = ulSMBusHWReadIPVersion( pxSMBusProfile );

                pxSMBusVersion->usIpVerMajor = ( uint16_t )( ( ulIpVersion & 0xFFFF0000 ) >> 16 );
                pxSMBusVersion->usIpVerMinor = ( uint16_t )( ulIpVersion & 0x0000FFFF );

                xError = SMBUS_SUCCESS;
            }
        }
        else
        {
            pxSMBusVersion->usIpVerMajor = 0;
            pxSMBusVersion->usIpVerMinor = 0;
        }
    }

    return ( xError );
#endif
}

/****************************************************************************
*
* @brief    Disables and then clears all SMBus interrupts
*
*****************************************************************************/
SMBus_Error_Type xSMBusInterruptDisableAndClearInterrupts( struct SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    SMBus_Error_Type xError = SMBUS_ERROR;

    if( NULL != pxSMBusProfile )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
            {
                vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                SMBUS_ERROR, __LINE__ );
            }
            else
            {
            /* Disable all Interrupts */
            vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 0 );
            vSMBusHWWriteIRQIER( pxSMBusProfile, 0 );
            vSMBusHWWriteERRIRQIER( pxSMBusProfile, 0 );

            /* Clear Interrupts - Write 1 to clear */
            vSMBusHWWriteIRQISR( pxSMBusProfile, 0x0000FFFF );
            vSMBusHWWriteERRIRQISR( pxSMBusProfile, 0x000FFFFF );

            xError = SMBUS_SUCCESS;
        }
    }

    return ( xError );
}

/*******************************************************************************
*
* @brief    enables all necessary SMBus interrupts
*
*******************************************************************************/
SMBus_Error_Type xSMBusInterruptEnableInterrupts( struct SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    SMBus_Error_Type xError = SMBUS_ERROR;

    if( NULL != pxSMBusProfile )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                            SMBUS_ERROR, __LINE__ );
        }
        else
        {
            /* Enable all Interrupts */
            vSMBusHWWriteIRQIER( pxSMBusProfile, 0x0000DFEF );
            vSMBusHWWriteERRIRQIER( pxSMBusProfile, 0x000FFFFF );
            vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 1 );

            xError = SMBUS_SUCCESS;
        }
    }

    return ( xError );
}

/*******************************************************************************
*
* @brief    Checks hardware is present at the supplied base address
*           Sets up hardware registers for the frequency class supplied
*           initializes software structures, sets up log and event queues
*
*****************************************************************************/
SMBus_Error_Type xInitSMBus( struct SMBUS_PROFILE_TYPE** ppxSMBusProfile, SMBus_Freq_Class_Type xFrequencyClass, void * pvBaseAddress,
                SMBUS_LOG_LEVEL_TYPE xLogLevel, SMBUS_USER_SUPPLIED_ENVIRONMENT_READ_TICKS pFnReadTicks )
{
    SMBus_Error_Type xError                         = SMBUS_SUCCESS;
    int              i                              = 0;
    int              j                              = 0;
    uint32_t         ulAXIClockFrequency            = 0;
    uint32_t         ulPHYInputGlitchFilterEnable   = 0;
    uint32_t         ulPHYInputGlitchFilterDuration = 0;
    uint32_t         ulConstant                     = 0;

    /* NOTE: pFnReadTicks can be NULL. It will be checked in the
             logging function where it is used */
    if( ( NULL != ppxSMBusProfile )  &&
        ( NULL == *ppxSMBusProfile )  &&
        ( NULL != pvBaseAddress )  &&
        ( SMBUS_FREQ_MAX > xFrequencyClass )  &&
        ( SMBUS_LOG_LEVEL_MAX > xLogLevel ) )
    {
        *ppxSMBusProfile = &xSMBusProfile;

        /* Check if we have initialized using this profile already */
        if( SMBUS_INITIALIZATION_CODE != (*ppxSMBusProfile)->ulInitialize )
        {
            /* Circular Event Log Initialize */
            vLogInitialize( *ppxSMBusProfile );
            (*ppxSMBusProfile)->pFnReadTicks = pFnReadTicks;
            (*ppxSMBusProfile)->pvBaseAddress = pvBaseAddress;
            (*ppxSMBusProfile)->xLogLevel = xLogLevel;

            uint32_t version  = ulSMBusHWReadIPVersion( *ppxSMBusProfile );
            vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO,
                            SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG, version, __LINE__ );

            if( SMBUS_MAGIC_NUMBER == ulSMBusHWReadIPMagicNum( *ppxSMBusProfile ) )
            {
                ulAXIClockFrequency = ulSMBusHWReadBuildConfig0( *ppxSMBusProfile );
                ulPHYInputGlitchFilterEnable = ulSMBusHWReadPHYFilterControlEnable( *ppxSMBusProfile );

                if( 1 == ulPHYInputGlitchFilterEnable )
                {
                    ulPHYInputGlitchFilterDuration =
                        SMBUS_GET_GLITCH_FILTER_DUR( ulSMBusHWReadPHYFilterControlDuration( *ppxSMBusProfile ) );
                    ulConstant = SMBUS_GET_CONSTANT_WITH_GLITCH( ulPHYInputGlitchFilterDuration );
                }
                else
                {
                    ulConstant = SMBUS_GET_CONSTANT;
                }

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TBUF_MIN_100KHZ, ulAXIClockFrequency ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TBUF_MIN_400KHZ, ulAXIClockFrequency ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TBUF_MIN_1MHZ, ulAXIClockFrequency ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TSU_DAT_MIN_100KHZ, ulAXIClockFrequency ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TSU_DAT_MIN_400KHZ, ulAXIClockFrequency ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TSU_DAT_MIN_1MHZ, ulAXIClockFrequency ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_TGT_DATA_HOLD_100KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TGT_DATA_HOLD_400KHZ, ulAXIClockFrequency ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_TGT_DATA_HOLD_1MHZ, ulAXIClockFrequency ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_DATA_HOLD_100KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_CTLR_DATA_HOLD_400KHZ, ulAXIClockFrequency ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE( SMBUS_CTLR_DATA_HOLD_1MHZ, ulAXIClockFrequency ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_HOLD_100KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_HOLD_400KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_HOLD_1MHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_SETUP_100KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_SETUP_400KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_SETUP_1MHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_STOP_SETUP_100KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_STOP_SETUP_400KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_STOP_SETUP_1MHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_LOW_100KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_LOW_400KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_LOW_1MHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );

                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_HIGH_100KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_HIGH_400KHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );
                vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                            SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_HIGH_1MHZ, ulAXIClockFrequency, ulConstant ), __LINE__ );

                switch( xFrequencyClass )
                {
                case SMBUS_FREQ_100KHZ:
                    vSMBusHWWritePHYBusFreetime( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TBUF_MIN_100KHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYTgtDataSetupTgtDataSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TSU_DAT_MIN_100KHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYTgtDataHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_TGT_DATA_HOLD_100KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlDataHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_DATA_HOLD_100KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlStartHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_HOLD_100KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlStartSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_SETUP_100KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlStopSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_STOP_SETUP_100KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlClkTLow( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_LOW_100KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlClkTHigh( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_HIGH_100KHZ, ulAXIClockFrequency, ulConstant ) );
                    break;

                case SMBUS_FREQ_400KHZ:
                    vSMBusHWWritePHYBusFreetime( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TBUF_MIN_400KHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYTgtDataSetupTgtDataSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TSU_DAT_MIN_400KHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYTgtDataHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TGT_DATA_HOLD_400KHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYCtrlDataHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_CTLR_DATA_HOLD_400KHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYCtrlStartHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_HOLD_400KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlStartSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_SETUP_400KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlStopSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_STOP_SETUP_400KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlClkTLow( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_LOW_400KHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlClkTHigh( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_HIGH_400KHZ, ulAXIClockFrequency, ulConstant ) );
                    break;

                case SMBUS_FREQ_1MHZ:
                    vSMBusHWWritePHYBusFreetime( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TBUF_MIN_1MHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYTgtDataSetupTgtDataSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TSU_DAT_MIN_1MHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYTgtDataHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_TGT_DATA_HOLD_1MHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYCtrlDataHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE( SMBUS_CTLR_DATA_HOLD_1MHZ, ulAXIClockFrequency ) );
                    vSMBusHWWritePHYCtrlStartHold( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_HOLD_1MHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlStartSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_START_SETUP_1MHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlStopSetup( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_STOP_SETUP_1MHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlClkTLow( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_LOW_1MHZ, ulAXIClockFrequency, ulConstant ) );
                    vSMBusHWWritePHYCtrlClkTHigh( *ppxSMBusProfile,
                        SMBUS_GET_FREQUENCY_VALUE_MINUS_CONSTANT( SMBUS_CTLR_CLK_HIGH_1MHZ, ulAXIClockFrequency, ulConstant ) );
                    break;

                default:
                    xError = SMBUS_ERROR;
                    break;
                }

                if( SMBUS_SUCCESS == xError )
                {
                    vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO,
                                    SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                                    ulSMBusHWReadBuildConfig0( *ppxSMBusProfile ), __LINE__ );
                    vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_INFO,
                                    SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_DEBUG,
                                    ulSMBusHWReadBuildConfig1( *ppxSMBusProfile ), __LINE__ );

                    (*ppxSMBusProfile)->ucInstanceInPlay = SMBUS_INVALID_INSTANCE;
                    (*ppxSMBusProfile)->ulTransactionID = 0;

                    for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_INSTANCES; i++ )
                    {
                        (*ppxSMBusProfile)->xSMBusInstance[i].ulFirewall1            = SMBUS_FIREWALL1;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ulFirewall2            = SMBUS_FIREWALL2;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ulFirewall3            = SMBUS_FIREWALL3;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ucInstanceInUse        = SMBUS_FALSE;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ucSMBusAddress         = 0;
                        (*ppxSMBusProfile)->xSMBusInstance[i].pFnGetProtocol         = NULL;
                        (*ppxSMBusProfile)->xSMBusInstance[i].pFnGetData             = NULL;
                        (*ppxSMBusProfile)->xSMBusInstance[i].pFnWriteData           = NULL;
                        (*ppxSMBusProfile)->xSMBusInstance[i].pFnAnnounceResult      = NULL;
                        (*ppxSMBusProfile)->xSMBusInstance[i].pFnArpAddressChange    = NULL;
                        (*ppxSMBusProfile)->xSMBusInstance[i].pFnBusError            = NULL;
                        (*ppxSMBusProfile)->xSMBusInstance[i].pFnBusWarning          = NULL;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ucPECRequired          = SMBUS_FALSE;
                        (*ppxSMBusProfile)->xSMBusInstance[i].xARPCapability         = SMBUS_ARP_CAPABILITY_UNKNOWN;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ucARFlag               = SMBUS_FALSE;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ucAVFlag               = SMBUS_FALSE;
                        (*ppxSMBusProfile)->xSMBusInstance[i].xProtocol              = SMBUS_PROTOCOL_NONE;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ucThisInstanceNumber   = SMBUS_INVALID_INSTANCE;
                        (*ppxSMBusProfile)->xSMBusInstance[i].ucUDIDMatchedInstance  = SMBUS_INVALID_INSTANCE;

                        vEventBufferInitialize( &( (*ppxSMBusProfile)->xSMBusInstance[i].xEventSourceCircularBuffer ),
                                                (*ppxSMBusProfile)->xSMBusInstance[i].xCircularBuffer, SMBUS_MAX_EVENT_ELEMENTS );

                        /* Add a pointer back to the top level */
                        (*ppxSMBusProfile)->xSMBusInstance[i].pxSMBusProfile = *ppxSMBusProfile;

                        /* Reset logs */
                        for( j = 0; j < SMBUS_PROTOCOL_NONE; j++ )
                        {
                            (*ppxSMBusProfile)->xSMBusInstance[i].ulMessagesComplete[j] = 0;
                            (*ppxSMBusProfile)->xSMBusInstance[i].ulMessagesInitiated[j] = 0;
                        }
                    }
                    vSMBusHWWriteCtrlDescFifoReset( *ppxSMBusProfile, 1 );
                    vSMBusHWWriteTgtRxFifoReset( *ppxSMBusProfile, 1 );
                    vSMBusHWWriteCtrlRxFifoReset( *ppxSMBusProfile, 1 );
                }
                (*ppxSMBusProfile)->ulInitialize = SMBUS_INITIALIZATION_CODE;
            }
            else
            {
                xError = SMBUS_ERROR;
            }
        }
        else
        {
            xError = SMBUS_ERROR;
        }
    }
    else
    {
        xError = SMBUS_ERROR;
    }

    return ( xError );
}

/*******************************************************************************
*
* @brief    Checks all instances have already been removed
*           If so sets Profile structure to default values
*
*****************************************************************************/
SMBus_Error_Type xDeinitSMBus( struct SMBUS_PROFILE_TYPE** ppxSMBusProfile )
{
    SMBus_Error_Type xError = SMBUS_SUCCESS;
    int              i      = 0;

    if( ( NULL != ppxSMBusProfile )  &&
        ( NULL != *ppxSMBusProfile ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( *ppxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( *ppxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_INSTANCES; i++ )
            {
                if( SMBUS_TRUE == (*ppxSMBusProfile)->xSMBusInstance[i].ucInstanceInUse )
                {
                    xError = SMBUS_ERROR;
                    break;
                }
            }
        }

        if( SMBUS_SUCCESS == xError )
        {
            xSMBusInterruptDisableAndClearInterrupts( *ppxSMBusProfile );
            (*ppxSMBusProfile)->pvBaseAddress   = NULL;
            (*ppxSMBusProfile)->pFnReadTicks    = NULL;
            (*ppxSMBusProfile)->ulInitialize    = SMBUS_DEINITIALIZATION_CODE;
            *ppxSMBusProfile                    = NULL;
        }
    }
    else
    {
        xError = SMBUS_ERROR;
    }

    return ( xError );
}

/*******************************************************************************
*
* @brief    Checks that a free instance slot is available and if so stores the
*           supplied data associated with the instance and enables the hardware
*           to send or receive SMBus messages for the supplied instance
*
*******************************************************************************/
uint8_t ucCreateSMBusInstance( struct SMBUS_PROFILE_TYPE* pxSMBusProfile,
                               uint8_t ucSMBusAddress,
                               uint8_t ucUDID[SMBUS_UDID_LENGTH],
                               SMBus_ARP_Capability xARPCapability,
                               SMBUS_USER_SUPPLIED_ENVIRONMENT_GET_PROTOCOL_TYPE pFnGetProtocol,
                               SMBUS_USER_SUPPLIED_ENVIRONMENT_GET_DATA_TYPE pFnGetData,
                               SMBUS_USER_SUPPLIED_ENVIRONMENT_WRITE_DATA_TYPE pFnWriteData,
                               SMBUS_USER_SUPPLIED_ENVIRONMENT_COMMAND_COMPLETE pFnAnnounceResult,
                               SMBUS_USER_SUPPLIED_ENVIRONMENT_ARP_ADRRESS_CHANGE pFnArpAddressChange,
                               SMBUS_USER_SUPPLIED_ENVIRONMENT_BUS_ERROR pFnBusError,
                               SMBUS_USER_SUPPLIED_ENVIRONMENT_BUS_WARNING pFnBusWarning,
                               uint8_t  ucSimpleDevice)
{
    uint8_t ucInstanceToReturn = SMBUS_INVALID_INSTANCE;
    uint8_t ucOkToContinue     = SMBUS_TRUE;
    int     i                  = 0;
    int     j                  = 0;

    if( ( NULL != pxSMBusProfile ) &&
        ( ( NULL != pFnGetProtocol )  || ( SMBUS_TRUE == ucSimpleDevice ) ) &&
        ( NULL !=  pFnGetData ) &&
        ( NULL !=  pFnWriteData ) &&
        ( NULL !=  pFnAnnounceResult ) )
    {

        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            ucOkToContinue = SMBUS_FALSE;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            /* Pre-checks before allowing the instance to be added */
            if( ( SMBUS_ARP_CAPABILITY_UNKNOWN == xARPCapability ) ||
                ( SMBUS_ARP_NON_ARP_CAPABLE < xARPCapability) )
            {
                ucOkToContinue = SMBUS_FALSE;
            }

            if( ( SMBUS_INVALID_ADDRESS_MASK & ucSMBusAddress ) &&
                ( SMBUS_ARP_CAPABLE == xARPCapability )         &&
                ( SMBUS_UDID_DYNAMIC_AND_PERSISTENT != ( ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE] & SMBUS_UDID_ADDRESS_TYPE_MASK ) ) )
            {
                ucOkToContinue = SMBUS_FALSE;
            }

            if( ( SMBUS_ARP_CAPABLE == xARPCapability ) &&
                ( SMBUS_UDID_FIXED_ADDRESS == ( ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE] & SMBUS_UDID_ADDRESS_TYPE_MASK ) ) )
            {
                ucOkToContinue = SMBUS_FALSE;
            }
        }

        if( SMBUS_TRUE == ucOkToContinue )
        {
            if( SMBUS_ARP_CAPABLE != xARPCapability )
            {
                for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
                {
                    if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
                    {
                        /* Check that the address is not already being used */
                        if( pxSMBusProfile->xSMBusInstance[i].ucSMBusAddress  == ucSMBusAddress )
                        {
                            ucOkToContinue = SMBUS_FALSE;
                            break;
                        }
                    }
                }
            }
        }

        if( SMBUS_TRUE == ucOkToContinue )
        {
            for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
            {
                if( SMBUS_FALSE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
                {
                    pxSMBusProfile->xSMBusInstance[i].ucSMBusAddress                = ucSMBusAddress;
                    pxSMBusProfile->xSMBusInstance[i].pFnGetProtocol                = pFnGetProtocol;
                    pxSMBusProfile->xSMBusInstance[i].pFnGetData                    = pFnGetData;
                    pxSMBusProfile->xSMBusInstance[i].pFnWriteData                  = pFnWriteData;
                    pxSMBusProfile->xSMBusInstance[i].pFnAnnounceResult             = pFnAnnounceResult;
                    pxSMBusProfile->xSMBusInstance[i].pFnArpAddressChange           = pFnArpAddressChange;
                    pxSMBusProfile->xSMBusInstance[i].pFnI2CGetData                 = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnI2CWriteData               = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnI2CAnnounceResult          = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnBusError                   = pFnBusError;
                    pxSMBusProfile->xSMBusInstance[i].pFnBusWarning                 = pFnBusWarning;
                    pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse               = SMBUS_TRUE;
                    pxSMBusProfile->xSMBusInstance[i].ucTriggerFSM                  = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[i].xARPCapability                = xARPCapability;
                    /* TODO Check correct initial values depending on ARP capability */
                    pxSMBusProfile->xSMBusInstance[i].ucARFlag                      = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[i].ulI2CDevice                   = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[i].ucFifoEmptyWhileInDoneCount   = 0;

                    if( SMBUS_ARP_CAPABLE == xARPCapability )
                    {
                        /* AV Flag should be read from NVRAM in the case of Dynamic & Persistent Target Address */
                        if( SMBUS_UDID_DYNAMIC_AND_PERSISTENT == ( ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE] & SMBUS_UDID_ADDRESS_TYPE_MASK ) )
                        {
                            /* We are deciding the AV Flag from the address used */
                            /* If the address is valid ie below 0x80 then set flag */
                            /* If the address is invalid ie 0x80 or above then clear flag */
                            /* If the address is equal to 0x00 then clear flag */
                            if( ( SMBUS_INVALID_ADDRESS_MASK & ucSMBusAddress ) || ( 0 == ucSMBusAddress ) )
                            {
                                /* Invalid address hence ARP needed to set both address and AV Flag */
                                pxSMBusProfile->xSMBusInstance[i].ucAVFlag          = SMBUS_FALSE;
                            }
                            else
                            {
                                pxSMBusProfile->xSMBusInstance[i].ucAVFlag          = SMBUS_TRUE;
                            }
                        }
                        else
                        {
                            pxSMBusProfile->xSMBusInstance[i].ucAVFlag          = SMBUS_FALSE;
                        }
                    }
                    else
                    {
                        pxSMBusProfile->xSMBusInstance[i].ucAVFlag              = SMBUS_TRUE;
                    }

                    pxSMBusProfile->xSMBusInstance[i].ucNewDeviceSlaveAddress   = 0;
                    pxSMBusProfile->xSMBusInstance[i].ucNackSent                = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[i].ucSimpleDevice            = ucSimpleDevice;
                    pxSMBusProfile->xSMBusInstance[i].ucThisInstanceNumber      = i;

                    for( j = 0; j < SMBUS_UDID_LENGTH; j++ )
                    {
                        pxSMBusProfile->xSMBusInstance[i].ucUDID[j] = ucUDID[j];
                    }

                    /* PEC Supported is taken from the PEC SUpported bit within the UDID */
                    pxSMBusProfile->xSMBusInstance[i].ucPECRequired =
                        ( ucUDID[SMBUS_UDID_DEVICE_CAPABILITIES_BYTE] & SMBUS_UDID_PEC_SUPPORTED_BIT );

                    vSMBusHWWriteTgtControlAddress( pxSMBusProfile, i, ucSMBusAddress );

                    /* Only non-ARP capable instances should enable the bus at this point */
                    /* Non-fixed ARP-capable instances should wait on the ARP process completing before enabling the bus */
                    if( SMBUS_ARP_CAPABLE != pxSMBusProfile->xSMBusInstance[i].xARPCapability )
                    {
                        vSMBusHWWriteTgtControlEnable( pxSMBusProfile, i, 1 );
                    }
                    ucInstanceToReturn = i;

                    /* If an ARP instance hasn't already been added - add it now */
                    if( SMBUS_FALSE == pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucInstanceInUse )
                    {
                        /* Don't add ARP instance if the created device is non-ARP capable */
                        if( SMBUS_ARP_NON_ARP_CAPABLE != xARPCapability )
                        {
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucInstanceInUse         = SMBUS_TRUE;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucSMBusAddress          = SMBUS_DEVICE_DEFAULT_ARP_ADDRESS;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucPECRequired           = SMBUS_TRUE;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucThisInstanceNumber    = SMBUS_ARP_INSTANCE_ID;
                        vSMBusHWWriteTgtControlAddress( pxSMBusProfile,
                                                            SMBUS_ARP_INSTANCE_ID, SMBUS_DEVICE_DEFAULT_ARP_ADDRESS );
                        vSMBusHWWriteTgtControlEnable( pxSMBusProfile, SMBUS_ARP_INSTANCE_ID, 1 );
                        }
                    }
                    break;
                }
            }
        }
    }

    return ucInstanceToReturn;
}

/******************************************************************************
*
* @brief    Checks that the supplied instance is present and attempts to remove
*           it. If the instance being removed is the only instance then the ARP
*           instance is also removed
*
*****************************************************************************/
SMBus_Error_Type xDestroySMBusInstance( struct SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucSMBusInstanceID )
{
    SMBus_Error_Type xError               = SMBUS_ERROR;
    uint8_t          ucDestroyArpInstance = SMBUS_TRUE;
    int              i                    = 0;
    int              j                    = 0;

    if( ( NULL != pxSMBusProfile ) &&
        ( SMBUS_LAST_NON_ARP_SMBUS_INSTANCE >= ucSMBusInstanceID ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ucInstanceInUse )
            {
                if( SMBUS_STATE_INITIAL == pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].xState )
                {
                    vSMBusHWWriteTgtControlEnable( pxSMBusProfile, ucSMBusInstanceID, 0 );
                    vSMBusHWWriteTgtControlAddress( pxSMBusProfile, ucSMBusInstanceID, 0 );
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ulFirewall1           = SMBUS_FIREWALL1;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ulFirewall2           = SMBUS_FIREWALL2;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ulFirewall3           = SMBUS_FIREWALL3;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ucInstanceInUse       = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ucSMBusAddress        = 0;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pFnGetProtocol        = NULL;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pFnGetData            = NULL;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pFnWriteData          = NULL;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pFnAnnounceResult     = NULL;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pFnArpAddressChange   = NULL;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pFnBusError           = NULL;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pFnBusWarning         = NULL;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ucPECRequired         = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].xARPCapability        = SMBUS_ARP_CAPABILITY_UNKNOWN;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ucARFlag              = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ucAVFlag              = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].xProtocol             = SMBUS_PROTOCOL_NONE;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ucThisInstanceNumber  = SMBUS_INVALID_INSTANCE;
                    vEventBufferInitialize( &( pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].xEventSourceCircularBuffer ),
                                            pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].xCircularBuffer,
                                            SMBUS_MAX_EVENT_ELEMENTS );

                    /* Add a pointer back to the top level */
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].pxSMBusProfile = pxSMBusProfile;

                    /* Reset logs */
                    for( j = 0; j < SMBUS_PROTOCOL_NONE; j++ )
                    {
                        pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ulMessagesComplete[j]     = 0;
                        pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID].ulMessagesInitiated[j]    = 0;
                    }

                    /* Now check if ARP needs to be removed */
                    for( i = 0; i < ( SMBUS_NUMBER_OF_SMBUS_INSTANCES - 1 ); i++ )
                    {
                        /* If any instance is still in use then leave the ARP instance intact */
                        if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
                        {
                            if( SMBUS_ARP_NON_ARP_CAPABLE != pxSMBusProfile->xSMBusInstance[i].xARPCapability )
                            {
                                ucDestroyArpInstance = SMBUS_FALSE;
                            }
                        }
                    }

                    if( SMBUS_TRUE == ucDestroyArpInstance )
                    {
                        vSMBusHWWriteTgtControlEnable( pxSMBusProfile, SMBUS_ARP_INSTANCE_ID, 0 );
                        vSMBusHWWriteTgtControlAddress( pxSMBusProfile, SMBUS_ARP_INSTANCE_ID, 0 );
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ulFirewall1             = SMBUS_FIREWALL1;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ulFirewall2             = SMBUS_FIREWALL2;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ulFirewall3             = SMBUS_FIREWALL3;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucInstanceInUse         = SMBUS_FALSE;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucSMBusAddress          = 0;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].pFnGetProtocol          = NULL;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].pFnGetData              = NULL;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucPECRequired           = SMBUS_FALSE;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].xARPCapability          = SMBUS_ARP_CAPABILITY_UNKNOWN;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucARFlag                = SMBUS_FALSE;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucAVFlag                = SMBUS_FALSE;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].xProtocol               = SMBUS_PROTOCOL_NONE;
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ucThisInstanceNumber    = SMBUS_INVALID_INSTANCE;
                        vEventBufferInitialize( &( pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].xEventSourceCircularBuffer ),
                                                pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].xCircularBuffer,
                                                SMBUS_MAX_EVENT_ELEMENTS );

                        /* Add a pointer back to the top level */
                        pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].pxSMBusProfile = pxSMBusProfile;

                        /* Reset logs */
                        for( j = 0; j < SMBUS_PROTOCOL_NONE; j++ )
                        {
                            pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ulMessagesComplete[j]   = 0;
                            pxSMBusProfile->xSMBusInstance[SMBUS_ARP_INSTANCE_ID].ulMessagesInitiated[j]  = 0;
                        }
                    }

                    xError = SMBUS_SUCCESS;
                }
            }
        }
    }

    return ( xError );
}

/*******************************************************************************
*
* @brief    Will initiate an SMBus message from the supplied intance as a
*           controller
*
*****************************************************************************/
SMBus_Error_Type xSMBusControllerInitiateCommand( struct SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucSMBusInstanceID,
                                        uint8_t ucSMBusDestinationAddress, uint8_t ucCommand,
                                        SMBus_Command_Protocol_Type xProtocol, uint16_t usDataSize, uint8_t* pucData,
                                        uint8_t ucPecRequiredForTransaction, uint32_t* pulTransactionID )
{
    SMBus_Error_Type xError = SMBUS_ERROR;

    if( ( NULL != pxSMBusProfile ) &&
        ( SMBUS_LAST_NON_ARP_SMBUS_INSTANCE >= ucSMBusInstanceID ) &&
        ( SMBUS_PROTOCOL_NONE > xProtocol ) &&
        ( SMBUS_DATA_SIZE_MAX >= usDataSize ) &&
        ( NULL != pucData ) &&
        ( SMBUS_TRUE >= ucPecRequiredForTransaction )  &&
        ( NULL != pulTransactionID ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else if( ( SMBUS_SMBCLK_LOW_TIMEOUT_DETECTED == ulSMBusHWReadPHYStatusSMBClkLowTimeout( pxSMBusProfile ) ) ||
                 ( SMBUS_SMBDAT_LOW_TIMEOUT_DETECTED == ulSMBusHWReadPHYStatusSMBDATLowTimeout( pxSMBusProfile ) ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            SMBUS_INSTANCE_TYPE* pxSMBusInstance = &( pxSMBusProfile->xSMBusInstance[ucSMBusInstanceID] );

            if( SMBUS_INVALID_INSTANCE == pxSMBusProfile->ucInstanceInPlay )
            {
                /* Do we need to check if one is in progress for this instance */
                if( SMBUS_STATE_INITIAL == pxSMBusInstance->xState )
                {
                    *pulTransactionID = pxSMBusProfile->ulTransactionID++;
                    pxSMBusProfile->ucInstanceInPlay = ucSMBusInstanceID;

                    /* Lets copy the data needed and the protocol to use */
                    memcpy( pxSMBusInstance->ucSendData, pucData, ( sizeof( uint8_t ) * usDataSize ) );
                    pxSMBusInstance->usSendDataSize = usDataSize;
                    pxSMBusInstance->usSendIndex = 0;
                    pxSMBusInstance->ucCommand = ucCommand;
                    pxSMBusInstance->xProtocol = xProtocol;

                    if( SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED == xProtocol )
                    {
                        pxSMBusInstance->ucCommand = ucCommand | SMBUS_ARP_UDID_DIRECTED_COMMAND;
                    }
                    if( SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED == xProtocol )
                    {
                        pxSMBusInstance->ucCommand = ucCommand & SMBUS_ARP_DIRECTED_COMMAND_ADDRESS_MASK;
                    }

                    pxSMBusInstance->ucSMBusDestinationAddress = ucSMBusDestinationAddress;
                    pxSMBusInstance->ucPecRequiredForTransaction = ucPecRequiredForTransaction;

                    /* First Reset the descriptor FIFO */
                    vSMBusHWWriteCtrlDescFifoReset( pxSMBusProfile, 1 );
                    vSMBusHWWriteCtrlRxFifoReset( pxSMBusProfile, 1 );

                    /* Create an action to get state machine into the next state; */
                    vSMBusGenerateEvent_E_SEND_NEXT_BYTE( pxSMBusInstance );

                    /* Disable interrupts */
                    vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 0 );

                    /* Call SMBusEventQueueHandle */
                    vSMBusEventQueueHandle( pxSMBusProfile );

                    /* Re-enable all the Controller interrupts */
                    vSMBusHWWriteIRQIER( pxSMBusProfile, 0x0000DFEF );

                    /* Re-enable interrupts */
                    vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 1 );

                    xError = SMBUS_SUCCESS;
                }
            }
        }
    }

    return ( xError );
}


/*******************************************************************************
*
* @brief    Retrieves SMBus log that is stored as a circular buffer in profile struct
*           as ASCII char array
*
*****************************************************************************/
SMBus_Error_Type xSMBusGetLog( struct SMBUS_PROFILE_TYPE* pxSMBusProfile, char* pcLogBuffer, uint32_t* pulLogSizeBytes )
{
    SMBus_Error_Type xError = SMBUS_ERROR;

    if( ( NULL != pxSMBusProfile ) &&
        ( NULL != pcLogBuffer ) &&
        ( NULL != pulLogSizeBytes ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            vLogDisplayLog( pxSMBusProfile, pcLogBuffer, pulLogSizeBytes );
            xError = SMBUS_SUCCESS;
        }
    }

    return ( xError );
}


/*******************************************************************************
*
* @brief    Resets SMBus Driver Log
*
*****************************************************************************/
SMBus_Error_Type xSMBusLogReset( struct SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    SMBus_Error_Type xError = SMBUS_ERROR;

    if( NULL != pxSMBusProfile )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            vLogInitialize( pxSMBusProfile );
            xError = SMBUS_SUCCESS;
        }
    }

    return ( xError );
}

/*******************************************************************************
*
* @brief    Enables logging
*
*******************************************************************************/
void vSMBusLogEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    if( NULL != pxSMBusProfile )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            pxSMBusProfile->xLogLevel = SMBUS_LOG_LEVEL_DEBUG;
        }
    }
}

/*******************************************************************************
*
* @brief    Disables logging
*
*******************************************************************************/
void vSMBusLogDisable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    if( NULL != pxSMBusProfile )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            pxSMBusProfile->xLogLevel = SMBUS_LOG_LEVEL_NONE;
        }
    }
}

/*******************************************************************************
*
* @brief    Resets the statistics log values for the specified instance
*
*******************************************************************************/
void vSMBusResetStatsLogInstance( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucSMBusInstance )
{
    int i = 0;

    if( ( NULL != pxSMBusProfile ) &&
        ( SMBUS_LAST_SMBUS_INSTANCE >= ucSMBusInstance ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            if( SMBUS_LAST_SMBUS_INSTANCE >= ucSMBusInstance )
            {
                for( i = 0; i < SMBUS_PROTOCOL_NONE; i++ )
                {
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstance].ulMessagesComplete[i] = 0;
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstance].ulMessagesInitiated[i] = 0;
                }
            }
        }
    }
}

/*******************************************************************************
*
* @brief    Reads the statistics log values for the specified instance
*
*******************************************************************************/
void vSMBusReadStatsLogInstance( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucSMBusInstance,
                                SMBUS_LOG_TYPE* pxSMBusMessageLog )
{
    int i = 0;

    if( ( NULL != pxSMBusProfile ) &&
        ( SMBUS_LAST_SMBUS_INSTANCE >= ucSMBusInstance )  &&
        ( NULL != pxSMBusMessageLog ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            for( i = 0; i < SMBUS_PROTOCOL_NONE; i++ )
            {
                pxSMBusMessageLog->ulMessagesComplete[i] =
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstance].ulMessagesComplete[i];
                pxSMBusMessageLog->ulMessagesInitiated[i] =
                    pxSMBusProfile->xSMBusInstance[ucSMBusInstance].ulMessagesInitiated[i];
            }
        }
    }
}

/*******************************************************************************
*
* @brief    Converts a protocol enum value to a text string for logging
*
*******************************************************************************/
char* pcProtocolToString( uint8_t ucProtocol )
{
    char* pcResult = prvpcProtocol_UNKNOWN;

    switch( ucProtocol )
    {
    case SMBUS_PROTOCOL_QUICK_COMMAND_HI:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_QUICK_COMMAND_HI;
        break;

    case SMBUS_PROTOCOL_QUICK_COMMAND_LO:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_QUICK_COMMAND_LO;
        break;

    case SMBUS_PROTOCOL_SEND_BYTE:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_SEND_BYTE;
        break;

    case SMBUS_PROTOCOL_RECEIVE_BYTE:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_RECEIVE_BYTE;
        break;

    case SMBUS_PROTOCOL_WRITE_BYTE:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_WRITE_BYTE;
        break;

    case SMBUS_PROTOCOL_WRITE_WORD:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_WRITE_WORD;
        break;

    case SMBUS_PROTOCOL_READ_BYTE:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_READ_BYTE;
        break;

    case SMBUS_PROTOCOL_READ_WORD:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_READ_WORD;
        break;

    case SMBUS_PROTOCOL_PROCESS_CALL:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_PROCESS_CALL;
        break;

    case SMBUS_PROTOCOL_BLOCK_WRITE:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_BLOCK_WRITE;
        break;

    case SMBUS_PROTOCOL_BLOCK_READ:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_BLOCK_READ;
        break;

    case SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_BLOCK_WRITE_BLOCK_READ_PROCESS_CALL;
        break;

    case SMBUS_PROTOCOL_HOST_NOTIFY:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_HOST_NOTIFY;
        break;

    case SMBUS_PROTOCOL_WRITE_32:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_WRITE_32;
        break;

    case SMBUS_PROTOCOL_READ_32:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_READ_32;
        break;

    case SMBUS_PROTOCOL_WRITE_64:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_WRITE_64;
        break;

    case SMBUS_PROTOCOL_READ_64:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_READ_64;
        break;

    case SMBUS_PROTOCOL_NONE:
        pcResult = prvpcProtocol_SMBUS_PROTOCOL_NONE;
        break;

    case SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP:
        pcResult = prvpcProtocol_SMBUS_ARP_PROTOCOL_PREPARE_TO_ARP;
        break;

    case SMBUS_ARP_PROTOCOL_RESET_DEVICE:
        pcResult = prvpcProtocol_SMBUS_ARP_PROTOCOL_RESET_DEVICE;
        break;

    case SMBUS_ARP_PROTOCOL_GET_UDID:
        pcResult = prvpcProtocol_SMBUS_ARP_PROTOCOL_GET_UDID;
        break;

    case SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS:
        pcResult = prvpcProtocol_SMBUS_ARP_PROTOCOL_ASSIGN_ADDRESS;
        break;

    case SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED:
        pcResult = prvpcProtocol_SMBUS_ARP_PROTOCOL_RESET_DEVICE_DIRECTED;
        break;

    case SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED:
        pcResult = prvpcProtocol_SMBUS_ARP_PROTOCOL_GET_UDID_DIRECTED;
        break;

    default:
        pcResult = prvpcProtocol_UNKNOWN;
        break;
    }

    return pcResult;
}

/*******************************************************************************
*
* @brief    Creates an i2c device to act as both a master and a slave
*           Checks that a free instance slot is available and if so stores the
*           supplied data associated with the instance and enables the hardware
*           to send or receive I2C messages for the supplied instance
*
*******************************************************************************/
uint8_t ucI2CCreateDevice( struct I2C_PROFILE_TYPE* pxI2cProfile,
							 uint8_t ucAddr,
							 I2C_USER_SUPPLIED_ENVIRONMENT_GET_DATA_TYPE    pFnGetData,
							 I2C_USER_SUPPLIED_ENVIRONMENT_WRITE_DATA_TYPE  pFnWriteData,
							 I2C_USER_SUPPLIED_ENVIRONMENT_COMMAND_COMPLETE	pFnAnnounceResult,
							 I2C_USER_SUPPLIED_ENVIRONMENT_BUS_ERROR		pFnBusError,
							 I2C_USER_SUPPLIED_ENVIRONMENT_BUS_WARNING		pFnBusWarning )
{
    uint8_t                     ucInstanceToReturn  = SMBUS_INVALID_INSTANCE;
    uint8_t                     ucOkToContinue      = SMBUS_TRUE;
    int                         i                   = 0;
    struct SMBUS_PROFILE_TYPE*  pxSMBusProfile      = NULL;

    if( ( NULL != pxI2cProfile ) &&
        ( NULL !=  pFnGetData ) &&
        ( NULL !=  pFnWriteData ) &&
        ( NULL !=  pFnAnnounceResult ) )
    {
        pxSMBusProfile = ( struct SMBUS_PROFILE_TYPE* )pxI2cProfile;

        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            ucOkToContinue = SMBUS_FALSE;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }

        if( SMBUS_INVALID_ADDRESS_MASK & ucAddr )
        {
            ucOkToContinue = SMBUS_FALSE;
        }

        if( SMBUS_TRUE == ucOkToContinue )
        {
            for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
            {
                if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
                {
                    /* Check that the address is not already being used */
                    if( pxSMBusProfile->xSMBusInstance[i].ucSMBusAddress  == ucAddr )
                    {
                        ucOkToContinue = SMBUS_FALSE;
                        break;
                    }
                }
            }
        }

        if( SMBUS_TRUE == ucOkToContinue )
        {
            for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_NON_ARP_INSTANCES; i++ )
            {
                if( SMBUS_FALSE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
                {
                    pxSMBusProfile->xSMBusInstance[i].ucSMBusAddress            = ucAddr;
                    pxSMBusProfile->xSMBusInstance[i].pFnGetProtocol            = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnGetData                = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnWriteData              = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnAnnounceResult         = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnArpAddressChange       = NULL;
                    pxSMBusProfile->xSMBusInstance[i].pFnI2CGetData             = pFnGetData;
                    pxSMBusProfile->xSMBusInstance[i].pFnI2CWriteData           = pFnWriteData;
                    pxSMBusProfile->xSMBusInstance[i].pFnI2CAnnounceResult      = pFnAnnounceResult;
                    pxSMBusProfile->xSMBusInstance[i].pFnBusError               = pFnBusError;
                    pxSMBusProfile->xSMBusInstance[i].pFnBusWarning             = pFnBusWarning;
                    pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse           = SMBUS_TRUE;
                    pxSMBusProfile->xSMBusInstance[i].ucTriggerFSM              = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[i].ulI2CDevice               = SMBUS_TRUE;

                    /* TODO Check correct initial values depending on ARP capability */
                    pxSMBusProfile->xSMBusInstance[i].ucNewDeviceSlaveAddress   = 0;
                    pxSMBusProfile->xSMBusInstance[i].ucNackSent                = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[i].ucSimpleDevice            = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[i].ucThisInstanceNumber      = i;

                    /* PEC not supported from driver*/
                    pxSMBusProfile->xSMBusInstance[i].ucPECRequired             = SMBUS_FALSE;
                    vSMBusHWWriteTgtControlAddress( pxSMBusProfile, i, ucAddr );

                    /* enable the bus */
                    vSMBusHWWriteTgtControlEnable( pxSMBusProfile, i, 1 );
                    ucInstanceToReturn = i;
                    break;
                }
            }
        }
    }

    return ucInstanceToReturn;
}

/******************************************************************************
*
* @brief    Destroys a previously created i2c device
*
*****************************************************************************/
uint8_t ucI2CDestroyDevice( struct I2C_PROFILE_TYPE* pxI2cProfile,
							  uint8_t ucDeviceId )
{
    SMBus_Error_Type            xError                  = SMBUS_ERROR;
    int                         j                       = 0;
    struct SMBUS_PROFILE_TYPE*  pxSMBusProfile          = NULL;


    if( ( NULL != pxI2cProfile ) &&
        ( SMBUS_LAST_NON_ARP_SMBUS_INSTANCE >= ucDeviceId ) )
    {
        pxSMBusProfile = ( struct SMBUS_PROFILE_TYPE* )pxI2cProfile;

        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[ucDeviceId].ucInstanceInUse )
            {
                if( SMBUS_STATE_INITIAL == pxSMBusProfile->xSMBusInstance[ucDeviceId].xState )
                {
                    vSMBusHWWriteTgtControlEnable( pxSMBusProfile, ucDeviceId, 0 );
                    vSMBusHWWriteTgtControlAddress( pxSMBusProfile, ucDeviceId, 0 );
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ulFirewall1           = SMBUS_FIREWALL1;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ulFirewall2           = SMBUS_FIREWALL2;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ulFirewall3           = SMBUS_FIREWALL3;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ucInstanceInUse       = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ucSMBusAddress        = 0;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].pFnI2CGetData         = NULL;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].pFnI2CWriteData       = NULL;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].pFnI2CAnnounceResult  = NULL;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].pFnBusError           = NULL;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].pFnBusWarning         = NULL;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ucPECRequired         = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].xARPCapability        = SMBUS_ARP_CAPABILITY_UNKNOWN;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ucARFlag              = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ucAVFlag              = SMBUS_FALSE;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].xProtocol             = SMBUS_PROTOCOL_NONE;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ucThisInstanceNumber  = SMBUS_INVALID_INSTANCE;
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].ulI2CDevice           = SMBUS_FALSE;
                    vEventBufferInitialize( &( pxSMBusProfile->xSMBusInstance[ucDeviceId].xEventSourceCircularBuffer ),
                                            pxSMBusProfile->xSMBusInstance[ucDeviceId].xCircularBuffer,
                                            SMBUS_MAX_EVENT_ELEMENTS );

                    /* Add a pointer back to the top level */
                    pxSMBusProfile->xSMBusInstance[ucDeviceId].pxSMBusProfile = pxSMBusProfile;

                    /* Reset logs */
                    for( j = 0; j < SMBUS_PROTOCOL_NONE; j++ )
                    {
                        pxSMBusProfile->xSMBusInstance[ucDeviceId].ulMessagesComplete[j]     = 0;
                        pxSMBusProfile->xSMBusInstance[ucDeviceId].ulMessagesInitiated[j]    = 0;
                    }

                    xError = SMBUS_SUCCESS;
                }
            }
        }
    }

    return ( xError );
}

/******************************************************************************
*
*	@brief	Writes data to a remote slave as a master
*
*******************************************************************************/
uint8_t ucI2CWriteData( struct I2C_PROFILE_TYPE* pxI2cProfile,
						  uint8_t  ucDeviceId,
						  uint8_t  ucAddr,
						  uint8_t* pucData,
						  uint16_t usNumBytes )
{
   SMBus_Error_Type xError = SMBUS_ERROR;
   struct SMBUS_PROFILE_TYPE*  pxSMBusProfile      = NULL;

    pxSMBusProfile = ( struct SMBUS_PROFILE_TYPE* )pxI2cProfile;

    if( ( NULL != pxSMBusProfile ) &&
        ( SMBUS_LAST_NON_ARP_SMBUS_INSTANCE >= ucDeviceId ) &&
        ( SMBUS_DATA_SIZE_MAX >= usNumBytes ) &&
        ( NULL != pucData ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            SMBUS_INSTANCE_TYPE* pxSMBusInstance = &( pxSMBusProfile->xSMBusInstance[ucDeviceId] );

            if( SMBUS_TRUE == pxSMBusInstance->ucInstanceInUse )
            {
                if( SMBUS_INVALID_INSTANCE == pxSMBusProfile->ucInstanceInPlay )
                {
                    /* Do we need to check if one is in progress for this instance */
                    if( SMBUS_STATE_INITIAL == pxSMBusInstance->xState )
                    {
                        /* Lets copy the data needed and the protocol to use */
                        memcpy( pxSMBusInstance->ucSendData, pucData, ( sizeof( uint8_t ) * usNumBytes ) );
                        pxSMBusProfile->ucInstanceInPlay            = ucDeviceId;
                        pxSMBusInstance->usSendDataSize             = usNumBytes;
                        pxSMBusInstance->usSendIndex                = 0;
                        pxSMBusInstance->ulI2CTransaction           = SMBUS_TRUE;
                        pxSMBusInstance->xProtocol                  = I2C_PROTOCOL_WRITE;
                        pxSMBusInstance->ucSMBusDestinationAddress  = ucAddr;

                        /* First Reset the descriptor FIFO */
                        vSMBusHWWriteCtrlDescFifoReset( pxSMBusProfile, 1 );
                        vSMBusHWWriteCtrlRxFifoReset( pxSMBusProfile, 1 );

                        /* Create an action to get state machine into the next state; */
                        vSMBusGenerateEvent_E_SEND_NEXT_BYTE( pxSMBusInstance );

                        /* Disable interrupts */
                        vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 0 );

                        /* Call SMBusEventQueueHandle */
                        vSMBusEventQueueHandle( pxSMBusProfile );

                        /* Re-enable all the Controller interrupts */
                        vSMBusHWWriteIRQIER( pxSMBusProfile, 0x0000DFEF );

                        /* Re-enable interrupts */
                        vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 1 );

                        xError = SMBUS_SUCCESS;
                    }
                }
            }
        }
    }

    return ( xError );
}

/******************************************************************************
*
*	@brief	Reads data from a remote slave as a master
*
*******************************************************************************/
uint8_t ucI2CReadData( struct I2C_PROFILE_TYPE* pxI2cProfile,
						 uint8_t   ucDeviceId,
						 uint8_t   ucAddr,
						 uint16_t  usNumBytes )
{
   SMBus_Error_Type xError = SMBUS_ERROR;
   struct SMBUS_PROFILE_TYPE*  pxSMBusProfile      = NULL;

    pxSMBusProfile = ( struct SMBUS_PROFILE_TYPE* )pxI2cProfile;

    if( ( NULL != pxSMBusProfile ) &&
        ( SMBUS_LAST_NON_ARP_SMBUS_INSTANCE >= ucDeviceId ) &&
        ( SMBUS_DATA_SIZE_MAX >= usNumBytes ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            SMBUS_INSTANCE_TYPE* pxSMBusInstance = &( pxSMBusProfile->xSMBusInstance[ucDeviceId] );

            if( SMBUS_INVALID_INSTANCE == pxSMBusProfile->ucInstanceInPlay )
            {
                /* Do we need to check if one is in progress for this instance */
                if( SMBUS_STATE_INITIAL == pxSMBusInstance->xState )
                {
                    pxSMBusProfile->ucInstanceInPlay            = ucDeviceId;
                    pxSMBusInstance->usExpectedByteCount        = usNumBytes;
                    pxSMBusInstance->xProtocol                  = I2C_PROTOCOL_READ;
                    pxSMBusInstance->ucSMBusDestinationAddress  = ucAddr;

                    /* First Reset the descriptor FIFO */
                    vSMBusHWWriteCtrlDescFifoReset( pxSMBusProfile, 1 );
                    vSMBusHWWriteCtrlRxFifoReset( pxSMBusProfile, 1 );

                    /* Create an action to get state machine into the next state; */
                    vSMBusGenerateEvent_E_SEND_NEXT_BYTE( pxSMBusInstance );

                    /* Disable interrupts */
                    vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 0 );

                    /* Call SMBusEventQueueHandle */
                    vSMBusEventQueueHandle( pxSMBusProfile );

                    /* Re-enable all the Controller interrupts */
                    vSMBusHWWriteIRQIER( pxSMBusProfile, 0x0000DFEF );

                    /* Re-enable interrupts */
                    vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 1 );

                    xError = SMBUS_SUCCESS;
                }
            }
        }
    }

    return ( xError );
}

/******************************************************************************
*
*	@brief	Writes data to and then reads from remote slave as a master
*
*******************************************************************************/
uint8_t ucI2CWriteReadData( struct I2C_PROFILE_TYPE* pxI2cProfile,
						 uint8_t   ucDeviceId,
						 uint8_t   ucAddr,
						 uint8_t*  pucWriteData,
						 uint16_t  usNumWriteBytes,
						 uint16_t  usNumReadBytes )
{
       SMBus_Error_Type xError = SMBUS_ERROR;
   struct SMBUS_PROFILE_TYPE*  pxSMBusProfile      = NULL;

    pxSMBusProfile = ( struct SMBUS_PROFILE_TYPE* )pxI2cProfile;

    if( ( NULL != pxSMBusProfile ) &&
        ( SMBUS_LAST_NON_ARP_SMBUS_INSTANCE >= ucDeviceId ) &&
        ( SMBUS_DATA_SIZE_MAX >= usNumWriteBytes ) &&
        ( NULL != pucWriteData ) &&
        ( SMBUS_DATA_SIZE_MAX >= usNumReadBytes ) &&
        ( I2C_READ_DATA_SIZE_MIN <= usNumReadBytes ) )
    {
        if( SMBUS_SUCCESS != xSMBusFirewallCheck( pxSMBusProfile ) )
        {
            xError = SMBUS_ERROR;
            vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_ERROR, SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_ERROR,
                                    SMBUS_ERROR, __LINE__ );
        }
        else
        {
            SMBUS_INSTANCE_TYPE* pxSMBusInstance = &( pxSMBusProfile->xSMBusInstance[ucDeviceId] );

            if( SMBUS_INVALID_INSTANCE == pxSMBusProfile->ucInstanceInPlay )
            {
                /* Do we need to check if one is in progress for this instance */
                if( SMBUS_STATE_INITIAL == pxSMBusInstance->xState )
                {
                    memcpy( pxSMBusInstance->ucSendData, pucWriteData, ( sizeof( uint8_t ) * usNumWriteBytes ) );
                    pxSMBusProfile->ucInstanceInPlay            = ucDeviceId;
                    pxSMBusInstance->usSendDataSize             = usNumWriteBytes;
                    pxSMBusInstance->usSendIndex                = 0;
                    pxSMBusInstance->usExpectedByteCount        = usNumReadBytes;
                    pxSMBusInstance->xProtocol                  = I2C_PROTOCOL_WRITE_READ;
                    pxSMBusInstance->ucSMBusDestinationAddress  = ucAddr;

                    /* First Reset the descriptor FIFO */
                    vSMBusHWWriteCtrlDescFifoReset( pxSMBusProfile, 1 );
                    vSMBusHWWriteCtrlRxFifoReset( pxSMBusProfile, 1 );

                    /* Create an action to get state machine into the next state; */
                    vSMBusGenerateEvent_E_SEND_NEXT_BYTE( pxSMBusInstance );

                    /* Disable interrupts */
                    vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 0 );

                    /* Call SMBusEventQueueHandle */
                    vSMBusEventQueueHandle( pxSMBusProfile );

                    /* Re-enable all the Controller interrupts */
                    vSMBusHWWriteIRQIER( pxSMBusProfile, 0x0000DFEF );

                    /* Re-enable interrupts */
                    vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 1 );

                    xError = SMBUS_SUCCESS;
                }
            }
        }
    }

    return ( xError );
}
